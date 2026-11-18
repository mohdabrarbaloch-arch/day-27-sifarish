# Sifarish · سفارش

A hybrid recommendation engine that shows its work.

Every recommendation comes with the signal that produced it and a sentence built
from the actual numbers. The blend between the three signals is not a fixed
ratio — it adapts to how much the engine knows about you, and to whether your
ratings actually carry information.

```
Item-item CF   42%   people who liked what you liked also liked this
User-user CF   14%   people like you liked this
Content        43%   this is about what you like
```

## What it does

- **215-title catalogue** across six genres, with a generated rating history
  (202 users, ~3,000 ratings) so the collaborative signals have something real
  to work with.
- **Three recommenders**, blended: item-item collaborative filtering, user-user
  collaborative filtering, and TF-IDF content similarity.
- **An adaptive blend.** Weights depend on rating count *and* rating spread.
- **Explanations that match the numbers.** Each card names the signal and, where
  there is one, the specific title that drove the match.
- **Offline evaluation against a baseline**, on a temporal holdout, in the admin
  view. Metrics without a baseline are decoration.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# generate a real secret:
python -c "import secrets; print(secrets.token_urlsafe(48))"

python -m app.seed          # 215 items, 202 users, ~3000 ratings
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000.

Demo accounts created by the seed:

| Role  | Email               | Password    |
|-------|---------------------|-------------|
| User  | demo@sifarish.pk    | demo12345   |
| Admin | admin@sifarish.pk   | admin12345  |

## Tests

```bash
pytest          # 53 tests
ruff check app tests
```

The tests assert on the mathematics, not on status codes alone. Several of them
encode bugs that were found and fixed — see *Bugs found* below.

## How the blend works

A fixed 0.5 / 0.3 / 0.2 blend is the most common reason a demo recommender falls
apart on a new user: 60% of the score comes from a collaborative signal that has
nothing to say. So the weights are a function of two things.

**1. How many ratings you have.** With zero, content gets everything. By ten,
item-item leads.

**2. How much your ratings vary.** This is the part most implementations miss.
Collaborative filtering works on *deviations from your own mean*. A user who
rates everything 5.0 has a standard deviation of zero, so every deviation is
zero, so item-item CF returns a flat vector of noise — no matter how many
ratings they have. Counting ratings alone would hand that user a 55% item-CF
weight and a feed that looks random. The blend detects the flat spread and hands
the weight to content, which does not care about your scale.

```python
blend_for_history(20, rating_std=1.0)  # item_cf 0.55, user_cf 0.25, content 0.20
blend_for_history(20, rating_std=0.0)  # item_cf 0.00, user_cf 0.00, content 1.00
```

## Evaluation

Leave-last-out temporal split: each user's most recent rating is held out, the
model is fitted on everything before it, and the held-out item is what we try to
rank. Users with a single rating are excluded rather than scored on data the
model has already seen.

| Metric             | Model  | Popularity baseline | Lift     |
|--------------------|--------|---------------------|----------|
| Precision@10       | 0.0303 | 0.0090              | +238.9%  |
| Recall@10          | 0.3035 | 0.0896              | +238.9%  |
| NDCG@10            | 0.1358 | 0.0400              | +239.1%  |
| Catalogue coverage | 0.9163 | 0.0698              | +1213.3% |

201 users evaluated, k = 10. Popularity bias: 24.8% of recommended slots go to
the top 10% of items, against 8.9% for the baseline — the model is more
diverse, not less.

Reproduce with `GET /api/admin/evaluate?k=10` as an admin.

## Bugs found while building this

These are in the README because a recommender demo that claims to be correct
without showing what went wrong is not worth reading.

**1. Min-max normalisation promoted noise to signal.** The obvious
`(x - min) / (max - min)` always maps the largest value to 1.0, however small it
is. A signal whose peak was 0.001 got stretched to the same range as one whose
peak was 0.9, so the blend weights stopped meaning anything. Fixed by scaling by
peak magnitude and multiplying by a confidence term that shrinks weak signals.

**2. A 0.02 similarity outranked everything.** An item with near-zero similarity
to every title a user had rated still produced a positive numerator, and dividing
by a denominator built from those same tiny numbers inflated it into the top
slot. A Romance title with ~0.02 similarity to every Thriller the demo account
had rated was scoring 0.97 at the top of a Thriller fan's feed. Fixed with a
similarity floor: sub-threshold similarities are zeroed, so "no evidence"
produces a zero score instead of a loud one.

**3. The explanation contradicted the label.** The card could say `item_cf`
while the sentence talked about content, because the branch was recomputed
independently. Fixed by passing the dominant source through to the explanation.

**4. "similarity 0.00" as evidence.** The explanation tested `> 0.0` and then
formatted to two decimals, so a similarity of 0.004 printed as "0.00" — a claim
the reader can see is false. The threshold now matches the printed precision.

**5. The blend summed to 99%.** Rounding each weight independently gave
42 + 14 + 43. Fixed with largest-remainder rounding, and a test asserts the
three shares sum to exactly 1.0 for every history size and spread.

**6. The feed went stale after rating.** Ratings are written from the catalogue,
so navigating back to the feed showed "no ratings yet" to a user who had just
rated three titles. Fixed by refetching on view entry.

**7. `hidden` lost to `display: grid`.** The modal backdrop was styled
`display: grid`, which outranks the user-agent `[hidden] { display: none }`, so
the invisible modal stayed on screen and swallowed every click.

## API

| Method | Path                              | Auth   | Purpose                          |
|--------|-----------------------------------|--------|----------------------------------|
| GET    | `/api/health`                     | —      | Liveness                         |
| POST   | `/api/auth/register`              | —      | Create account, returns a token  |
| POST   | `/api/auth/login`                 | —      | Sign in                          |
| GET    | `/api/auth/me`                    | Bearer | Current user                     |
| GET    | `/api/items`                      | —      | Catalogue, with search + filter  |
| GET    | `/api/items/{id}`                 | —      | One item                         |
| POST   | `/api/items`                      | Admin  | Create an item                   |
| GET    | `/api/ratings`                    | Bearer | Your ratings                     |
| PUT    | `/api/ratings`                    | Bearer | Create or update a rating        |
| DELETE | `/api/ratings/{item_id}`          | Bearer | Remove a rating                  |
| GET    | `/api/recommendations/me`         | Bearer | Personalised feed                |
| GET    | `/api/recommendations/cold-start` | —      | What a new visitor sees          |
| GET    | `/api/recommendations/for/{id}`   | Bearer | Another user's feed (self/admin) |
| GET    | `/api/admin/stats`                | Admin  | Catalogue and rating statistics  |
| GET    | `/api/admin/evaluate`             | Admin  | Model vs baseline                |

Interactive docs at `/docs` in development. They are disabled when
`ENVIRONMENT=production`.

## Security notes

- Passwords hashed with bcrypt; the hash never leaves the database.
- Login returns the same message and does the same work for "no such user" and
  "wrong password", so the form is not an account enumerator.
- Reading another user's recommendations returns **404**, not 403 — a 403 would
  confirm the account exists.
- `SECRET_KEY` has no usable default; the app refuses to start without one.
- Security headers on every response; CSP, `X-Frame-Options: DENY`, nosniff.
- Rate limiting is in-process and therefore per worker. Honest about it: N
  workers behind a load balancer allow N times the configured rate. A
  Redis-backed limiter is the production answer; the interface is the same shape
  so swapping it is a one-file change.

## What this is not

The catalogue and rating history are **generated**. The engine is not: it is the
same code that would run on real data, evaluated honestly against a baseline.
The seed models two layers of real audience behaviour — genre affinity and
sub-taste within a genre — because a purely random history gives the
collaborative filter nothing to recover and the evaluation would be meaningless.

## Layout

```
app/
  engine/          the recommender, independent of FastAPI and the database
    matrix.py      rating matrix, temporal splits
    similarity.py  adjusted cosine, shrinkage, top-k neighbours
    text.py        TF-IDF
    recommender.py the three signals and the adaptive blend
    evaluation.py  precision, recall, NDCG, coverage, popularity bias
  routers/         HTTP layer
  service.py       fits the engine from the database, with a cache
  seed.py          generated catalogue and rating history
web/               vanilla JS front end, no build step
```

The engine has no import from FastAPI or SQLAlchemy. That is deliberate: it is
testable on synthetic data, and the tests do exactly that.

## License

MIT.
