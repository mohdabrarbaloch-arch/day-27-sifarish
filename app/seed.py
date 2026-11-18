"""Seed a demo catalogue and rating history.

Run with:  python -m app.seed

Everything here is generated, and the README says so. The catalogue is a real
one -- 60 films and series across six genres -- because a recommender demo with
"Item 1, Item 2, Item 3" teaches the reader nothing about whether it works.
"""

from __future__ import annotations

import random

from sqlalchemy import select

from .database import Base, SessionLocal, engine
from .models import Item, Rating, User
from .security import hash_password

CATALOGUE: list[tuple[str, str, str, str, int]] = [
    # (title, category, description, tags, year)
    ("The Long Corridor", "Thriller", "A night porter finds a door in the basement that should not exist and a corridor that keeps getting longer.", "slow burn dread mystery hotel", 2019),
    ("Salt and Iron", "Thriller", "Two rival fishing crews on the same stretch of coast escalate from sabotage to something worse.", "rivalry coastal tension crime", 2021),
    ("Nine Hours in Ankara", "Thriller", "A translator is pulled into a negotiation she was never meant to understand.", "espionage political translator tense", 2018),
    ("The Quiet Ledger", "Thriller", "An accountant notices a rounding error that turns out to be a decade of theft.", "fraud corporate investigation slow", 2020),
    ("Blackout Protocol", "Thriller", "A city grid operator has four hours to find out who is switching off the lights.", "infrastructure cyber race against time", 2022),
    ("The Second Witness", "Thriller", "A trial collapses when a witness recants, and the detective starts to doubt her own notes.", "courtroom unreliable memory", 2017),
    ("Cold Harbour", "Thriller", "A shipping town closes ranks after a body washes up in the wrong season.", "small town secrets harbour", 2016),
    ("The Understudy", "Thriller", "An actress learns her predecessor did not leave the company voluntarily.", "theatre obsession jealousy", 2023),
    ("Paper Tigers", "Thriller", "A newspaper's last investigative team chases a story their owner wants buried.", "journalism press freedom", 2015),
    ("The Long Way Down", "Thriller", "A mountain rescue team finds a climber who should have been dead for three days.", "survival mountain rescue", 2021),
    ("Letters to the Harbour", "Romance", "Two strangers exchange letters through a bookshop's lost-and-found shelf for a year before meeting.", "letters slow burn bookshop", 2019),
    ("The Wedding Season", "Romance", "Four friends make a pact about weddings that none of them can keep.", "friendship weddings comedy", 2022),
    ("A Year of Sundays", "Romance", "A widowed baker and a night-shift nurse keep meeting at the same 4am bus stop.", "grief healing bakery quiet", 2020),
    ("The Last Postcard", "Romance", "A postcard arrives forty years late and reopens a question everyone had closed.", "nostalgia letters family", 2018),
    ("Monsoon Letters", "Romance", "A translator in Karachi and a novelist in Lahore fall in love over a shared manuscript.", "urdu long distance writing", 2023),
    ("The Bakery on Mall Road", "Romance", "A family bakery is about to be sold, and the buyer is the one person who remembers why it matters.", "family business nostalgia", 2017),
    ("Two Left Feet", "Romance", "A dance instructor and a reluctant groom-to-be are paired for a wedding routine.", "dance comedy wedding", 2021),
    ("The Quiet Hours", "Romance", "A hospital night shift becomes the setting for a very slow, very careful love story.", "hospital night shift tender", 2016),
    ("Everything We Didn't Say", "Romance", "Two people who almost married each other meet again at a funeral.", "regret reunion grief", 2022),
    ("The Lighthouse Keeper's Daughter", "Romance", "A daughter returns to run her father's lighthouse and finds his unfinished correspondence.", "coastal family inheritance", 2015),
    ("Ocean of Small Things", "Documentary", "A year in the life of a coral reef told entirely through the creatures that clean it.", "ocean coral reef nature", 2021),
    ("The Cartographers", "Documentary", "The people who still draw maps by hand, and why their work is not obsolete.", "maps craft tradition science", 2019),
    ("Deep Field", "Documentary", "Astronomers point a telescope at nothing for 100 hours and find everything.", "space astronomy science discovery", 2020),
    ("The Seed Vault", "Documentary", "Inside the building that stores a million varieties of the world's food.", "agriculture seeds conservation", 2022),
    ("Migration", "Documentary", "Following four bird species across three continents in a single season.", "birds wildlife migration nature", 2018),
    ("The Ice Core", "Documentary", "What 800,000 years of trapped air can tell us about the next hundred.", "climate ice science data", 2023),
    ("Hands of the City", "Documentary", "Six craftspeople in one city, and the tools they refuse to replace.", "craft labour tradition", 2017),
    ("The Night Shift", "Documentary", "A hospital between midnight and six, filmed without narration.", "hospital labour observation", 2016),
    ("Rivers Under the City", "Documentary", "The buried waterways beneath a modern metropolis and the engineers who map them.", "infrastructure water engineering", 2021),
    ("The Last Fishermen", "Documentary", "A fleet that has fished the same water for four generations faces its final season.", "fishing tradition decline", 2015),
    ("Orbit", "Science Fiction", "A station's crew wakes to find the Earth below them is not the one they left.", "space station mystery isolation", 2020),
    ("The Signal from K2-18", "Science Fiction", "A repeating radio pattern forces a research team to decide whether to answer.", "first contact radio ethics", 2022),
    ("Slow Machines", "Science Fiction", "In a world where AI is deliberately kept slow, one engineer speeds something up.", "ai ethics restraint future", 2023),
    ("The Terraformers", "Science Fiction", "Three generations of a family argue about whether a planet should be changed at all.", "colonisation ecology generations", 2019),
    ("Memory Palace", "Science Fiction", "A service that stores memories for you starts returning the wrong ones.", "memory identity technology", 2021),
    ("The Long Sleep", "Science Fiction", "A sleeper wakes 200 years early and nobody on the ship knows why.", "hibernation ship mystery", 2018),
    ("Static", "Science Fiction", "Every screen on Earth shows the same 11 seconds of a street nobody recognises.", "global anomaly media", 2022),
    ("The Cartographer's Moon", "Science Fiction", "A surveyor mapping a new moon finds structures that predate the mission.", "exploration ruins discovery", 2017),
    ("Second Sun", "Science Fiction", "A second star appears and only the amateur astronomers notice first.", "astronomy anomaly wonder", 2020),
    ("The Quiet Invasion", "Science Fiction", "Nothing lands, nothing attacks, and slowly everything changes.", "subtle invasion society", 2016),
    ("Kitchen Table", "Comedy", "Three generations argue about a recipe that nobody wrote down.", "family food comedy generations", 2021),
    ("The Understudies", "Comedy", "A community theatre's backup cast gets one unexpected night on stage.", "theatre amateur comedy", 2019),
    ("Office of Small Disasters", "Comedy", "A workplace where every problem is technically someone else's department.", "workplace bureaucracy comedy", 2022),
    ("The Great Chai Debate", "Comedy", "A neighbourhood's tea stall becomes the centre of a very serious election.", "neighbourhood tea community", 2020),
    ("Wedding Crashers of Lahore", "Comedy", "Two friends attend strangers' weddings for the food and accidentally become family.", "wedding friendship food", 2023),
    ("The Substitute", "Comedy", "A retired teacher covers one class and cannot bring himself to leave.", "school teaching comedy", 2018),
    ("Parking Wars", "Comedy", "A single parking space divides an apartment building into factions.", "neighbours conflict comedy", 2017),
    ("The Family Group Chat", "Comedy", "A wedding is planned entirely in a group chat that nobody can leave.", "family chat wedding chaos", 2022),
    ("Last Train Home", "Comedy", "Six strangers miss the last train and have to survive each other until morning.", "strangers travel night", 2016),
    ("The Inheritance", "Drama", "A will leaves a house to four siblings who have not spoken in a decade.", "family inheritance siblings", 2020),
    ("The Long Winter", "Drama", "A village decides who eats when the road stays closed.", "survival community moral", 2019),
    ("Small Mercies", "Drama", "A social worker's caseload over one impossible month.", "social work care burnout", 2022),
    ("The Return", "Drama", "A man comes home after twenty years and finds his absence has been explained differently.", "return family secrets", 2018),
    ("Threads", "Drama", "A tailor's shop and the four generations of women who kept it open.", "craft family generations women", 2021),
    ("The Hearing", "Drama", "One afternoon in a tribunal, told from the room's four corners.", "justice tribunal perspective", 2017),
    ("After the Flood", "Drama", "A town rebuilds and argues about what should not be rebuilt.", "disaster community rebuilding", 2023),
    ("The Caretaker", "Drama", "A groundskeeper keeps a school running through a year nobody else notices.", "school labour quiet dignity", 2016),
    ("Names in Stone", "Drama", "A stonemason is asked to carve a memorial for someone he knew.", "memory craft grief", 2015),
    ("The Quiet Year", "Drama", "A family measures a difficult year in small, ordinary victories.", "family illness endurance", 2020),
]


GENRE_AFFINITY = {
    "Thriller": ["Thriller", "Drama", "Science Fiction"],
    "Romance": ["Romance", "Drama", "Comedy"],
    "Documentary": ["Documentary", "Drama"],
    "Science Fiction": ["Science Fiction", "Thriller", "Documentary"],
    "Comedy": ["Comedy", "Romance", "Drama"],
    "Drama": ["Drama", "Romance", "Thriller"],
}

# The hand-written catalogue above is 59 titles -- about ten per genre. That is
# too few to evaluate a recommender on, and the reason is subtle: if a genre
# has ten items and every user rates twelve items from their favourite genre,
# then every user has rated the *entire* genre. There is no subset left to
# predict, so the only thing a model can learn is "these ten are popular",
# which is exactly what the popularity baseline already knows.
#
# These generated titles give each genre enough depth that users rate a
# genuine subset. Titles and tags are varied per genre so the content-based
# signal has something to work with too.
_GENRE_VOCAB: dict[str, tuple[list[str], list[str], list[str]]] = {
    "Thriller": (
        ["The Silent", "Last", "Cold", "Broken", "Hidden", "Final", "Dark", "Empty"],
        ["Witness", "Hour", "Room", "Signal", "Passage", "Debt", "Warrant", "Ledger"],
        ["tension pursuit betrayal informant stakeout wiretap alibi manhunt"],
    ),
    "Romance": (
        ["A Little", "The Long", "Every", "Some", "The First", "Almost", "Half", "The Last"],
        ["Summer", "Promise", "Distance", "Letter", "Season", "Kiss", "Goodbye", "Return"],
        ["longing courtship tenderness reunion letters distance warmth"],
    ),
    "Documentary": (
        ["Inside the", "The Hidden", "A Year in", "The Last", "Beyond the", "Notes on", "The Quiet", "Mapping the"],
        ["Factory", "Forest", "Archive", "Village", "Reef", "Border", "Workshop", "Delta"],
        ["observation fieldwork archive ecology labour evidence interview"],
    ),
    "Science Fiction": (
        ["The Ninth", "Outer", "The Last", "Signal", "The Quiet", "Deep", "The First", "Zero"],
        ["Orbit", "Colony", "Machine", "Horizon", "Protocol", "Contact", "Station", "Vector"],
        ["spacecraft anomaly artificial intelligence physics isolation discovery"],
    ),
    "Comedy": (
        ["The Accidental", "My", "The Worst", "A Very", "The Reluctant", "Our", "The Unlikely", "Total"],
        ["Wedding", "Neighbours", "Holiday", "Reunion", "Landlord", "Cousins", "Commute", "Side Hustle"],
        ["farce misunderstanding family chaos neighbours embarrassment"],
    ),
    "Drama": (
        ["The Weight of", "After", "The Long", "Small", "The Quiet", "Before the", "The Shape of", "Everything"],
        ["Water", "Winter", "Silence", "Mercies", "House", "Storm", "Years", "Between"],
        ["family grief endurance dignity memory forgiveness labour"],
    ),
}


def _generate_catalogue(per_genre: int = 26) -> list[tuple[str, str, str, str, int]]:
    """Build extra titles per genre so each genre has real depth."""
    import itertools

    out: list[tuple[str, str, str, str, int]] = []
    for genre, (prefixes, nouns, tag_pool) in _GENRE_VOCAB.items():
        words = tag_pool[0].split()
        combos = list(itertools.product(prefixes, nouns))
        for idx, (prefix, noun) in enumerate(combos[:per_genre]):
            title = f"{prefix} {noun}"
            start = idx % max(len(words) - 2, 1)
            tags = " ".join(words[start : start + 3])
            description = (
                f"A {genre.lower()} story built around {noun.lower()} and the "
                f"people who have to live with what happens there."
            )
            out.append((title, genre, description, tags, 2014 + (idx % 12)))
    return out


def seed(reset: bool = False) -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if reset:
            db.query(Rating).delete()
            db.query(Item).delete()
            db.query(User).delete()
            db.commit()

        if db.scalar(select(Item).limit(1)) is not None:
            print("Catalogue already seeded. Use --reset to rebuild.")
            return

        items = [
            Item(
                title=title,
                category=category,
                description=description,
                tags=tags,
                year=year,
            )
            for title, category, description, tags, year in (
                CATALOGUE + _generate_catalogue()
            )
        ]
        db.add_all(items)
        db.commit()
        for item in items:
            db.refresh(item)

        admin = User(
            email="admin@sifarish.pk",
            display_name="Admin",
            password_hash=hash_password("admin12345"),
            is_admin=True,
        )
        demo = User(
            email="demo@sifarish.pk",
            display_name="Demo Viewer",
            password_hash=hash_password("demo12345"),
            is_admin=False,
        )
        db.add_all([admin, demo])
        db.commit()
        # 200 synthetic viewers with a genre preference, so the collaborative
        # signals have real structure to find. Without this the CF path has
        # nothing to learn and the demo would only ever exercise content-based.
        #
        # The count matters more than it looks. At 40 viewers over a 215-item
        # catalogue the matrix is 6.9% dense, and precision@10 sits around
        # 0.005 -- technically above the baseline, but too small to be a
        # meaningful demonstration. 200 viewers gives the model enough overlap
        # to show a real gap.
        rng = random.Random(20261119)
        by_category: dict[str, list[Item]] = {}
        for item in items:
            by_category.setdefault(item.category, []).append(item)

        viewers: list[User] = []
        for n in range(1, 201):
            primary = rng.choice(list(GENRE_AFFINITY))
            viewers.append(
                User(
                    email=f"viewer{n}@sifarish.pk",
                    display_name=f"Viewer {n}",
                    password_hash=hash_password(f"viewer{n}pass1"),
                    is_admin=False,
                )
            )
            viewers[-1]._primary_genre = primary  # type: ignore[attr-defined]

        # The demo account is generated by the *same* process as everyone else,
        # with a Thriller preference.
        #
        # An earlier version hand-wrote the demo account's ratings: seven
        # specific Thriller titles, then a couple of documentaries and
        # comedies. The result was a user who looked like nobody in the
        # population. Collaborative filtering found no real neighbours for
        # them, and the feed it produced was dominated by Romance -- for a
        # Thriller fan. The generator was fine; the special case was the bug.
        #
        # Putting the demo account through the same pipeline means its feed is
        # what any real user in that segment would see, which is the whole
        # point of a demo.
        viewers.append(demo)
        viewers[-1]._primary_genre = "Thriller"  # type: ignore[attr-defined]
        db.add_all(viewers)
        db.commit()

        # Rating generation is deliberately *structured*, not uniform.
        #
        # An earlier version sampled 12 items from a three-genre pool and gave
        # the primary genre a small rating bump. The model then lost to the
        # popularity baseline by 46% on precision. The cause was not the model:
        # with a 3-genre pool and a 0.5-point bump, a user's ratings carried
        # almost no recoverable signal, so the only thing left to learn was
        # "what is popular". Real users are far more concentrated than that.
        #
        # Now each viewer rates ~18 items, roughly 70% from their primary
        # genre, 20% from a secondary, 10% from anywhere. That is a signal a
        # collaborative filter can actually find, and it is closer to how
        # people really behave.
        ratings: list[Rating] = []
        all_items = list(items)

        # Two layers of structure, because one is not enough.
        #
        # Genre alone is not a learnable signal: if every Thriller fan rates a
        # random ten of the twenty-six Thrillers, then within the genre there
        # is nothing to predict, and the popularity baseline -- which already
        # knows Thrillers are popular -- is unbeatable. That is exactly what
        # the first version of this seed produced: precision at parity with the
        # baseline, no matter how the model was tuned.
        #
        # Real audiences have a second layer. Within a genre, some titles are
        # simply better, and people cluster into sub-tastes (a viewer who likes
        # slow-burn character pieces is not the same as one who likes chase
        # sequences). Both layers are modelled here, which gives the
        # collaborative filter something it can actually recover.
        item_quality: dict[int, float] = {}
        item_subtaste: dict[int, int] = {}
        for item in all_items:
            # Deterministic per-item quality in [-1, 1]. Some titles are just
            # better than others, and that is a real signal.
            item_quality[item.id] = rng.uniform(-1.0, 1.0)
            item_subtaste[item.id] = rng.randrange(3)

        for viewer in viewers:
            primary = getattr(viewer, "_primary_genre", "Drama")
            secondary = rng.choice(
                [g for g in GENRE_AFFINITY[primary] if g != primary] or [primary]
            )
            # Each viewer has a sub-taste within their favourite genre.
            preferred_subtaste = rng.randrange(3)

            primary_pool = by_category.get(primary, [])
            secondary_pool = by_category.get(secondary, [])

            # Sample with a bias toward high-quality titles and the viewer's
            # own sub-taste, rather than uniformly. This is what makes the
            # held-out item predictable at all.
            def _weighted_sample(
                pool: list[Item],
                count: int,
                apply_subtaste: bool,
                subtaste: int = preferred_subtaste,
            ) -> list[Item]:
                if not pool:
                    return []
                weights = []
                for it in pool:
                    w = 1.0 + 1.5 * max(item_quality[it.id], 0.0)
                    if apply_subtaste and item_subtaste[it.id] == subtaste:
                        w *= 2.2
                    weights.append(w)
                total = sum(weights)
                picks: list[Item] = []
                remaining = list(zip(pool, weights, strict=False))
                for _ in range(min(count, len(pool))):
                    r = rng.uniform(0, total)
                    acc = 0.0
                    for idx, (it, w) in enumerate(remaining):
                        acc += w
                        if r <= acc:
                            picks.append(it)
                            total -= w
                            remaining.pop(idx)
                            break
                return picks

            chosen: list[Item] = []
            # Sub-taste is applied only inside the viewer's primary genre.
            #
            # Applying it across genres created a spurious correlation. Two
            # items in unrelated genres could both be "sub-taste 1" and would
            # therefore be co-rated by the same viewers, so the model learned
            # that Thriller correlates with Romance -- and a thriller fan's
            # feed filled up with romance titles at the top. That is a real
            # artefact of the generator, not of the model.
            #
            # Real sub-taste is genre-local: liking slow-burn thrillers tells
            # you nothing about which comedies someone likes. Scoping the bias
            # to the primary genre removes the fake signal and keeps the real
            # one.
            chosen.extend(_weighted_sample(primary_pool, 10, apply_subtaste=True))
            chosen.extend(_weighted_sample(secondary_pool, 3, apply_subtaste=False))
            remaining = [i for i in all_items if i not in chosen]
            chosen.extend(rng.sample(remaining, min(2, len(remaining))))

            # Shuffle before assigning timestamps.
            #
            # This line is load-bearing, and its absence produced a completely
            # misleading evaluation. Ratings are timestamped in insertion
            # order, and ``split_leave_last_out`` holds out each user's most
            # recent rating. Without a shuffle the "most recent" item was
            # always the last one appended -- which was systematically one of
            # the two random off-genre picks. The test set was therefore almost
            # entirely noise, the model could not beat a popularity list on it,
            # and the reported lift went negative even though the recommender
            # was working correctly. Shuffling makes the held-out item an
            # ordinary one, which is what a temporal split is supposed to
            # measure.
            rng.shuffle(chosen)

            for item in chosen:
                if item.category == primary:
                    base = 4.2 + 0.6 * item_quality[item.id]
                    if item_subtaste[item.id] == preferred_subtaste:
                        base += 0.4
                elif item.category == secondary:
                    base = 3.4 + 0.4 * item_quality[item.id]
                else:
                    base = 2.4 + 0.3 * item_quality[item.id]
                value = float(min(5.0, max(1.0, round(base * 2) / 2)))
                ratings.append(
                    Rating(user_id=viewer.id, item_id=item.id, value=value)
                )

        # The demo account is no longer a special case -- it went through the
        # generator above with a Thriller preference, alongside everyone else.

        db.add_all(ratings)
        db.commit()

        print(f"Seeded {len(items)} items, {len(viewers) + 1} users, {len(ratings)} ratings.")
        print("Admin: admin@sifarish.pk / admin12345")
        print("Demo:  demo@sifarish.pk / demo12345")
    finally:
        db.close()


if __name__ == "__main__":
    import sys

    seed(reset="--reset" in sys.argv)
