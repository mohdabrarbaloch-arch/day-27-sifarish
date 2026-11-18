"""Authentication endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import (
    find_user_by_email,
    get_current_user,
    login_limiter,
    rate_limit,
)
from ..models import User
from ..schemas import MessageOut, TokenOut, UserCreate, UserLogin, UserOut
from ..security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=TokenOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit(login_limiter))],
)
def register(payload: UserCreate, db: Session = Depends(get_db)) -> TokenOut:
    email = payload.email.lower()
    if find_user_by_email(db, email) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists",
        )

    user = User(
        email=email,
        display_name=payload.display_name,
        password_hash=hash_password(payload.password),
        is_admin=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(str(user.id))
    return TokenOut(access_token=token, user=UserOut.model_validate(user))


@router.post(
    "/login",
    response_model=TokenOut,
    dependencies=[Depends(rate_limit(login_limiter))],
)
def login(payload: UserLogin, db: Session = Depends(get_db)) -> TokenOut:
    user = find_user_by_email(db, payload.email.lower())

    # Same message and same work for "no such user" and "wrong password".
    # Distinguishing them turns the login form into an account enumerator.
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    token = create_access_token(str(user.id))
    return TokenOut(access_token=token, user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(user)


@router.post("/logout", response_model=MessageOut)
def logout() -> MessageOut:
    """Stateless JWTs cannot be revoked server-side without a denylist.

    The client drops the token. Saying so plainly beats pretending this
    endpoint does something it does not.
    """
    return MessageOut(detail="Signed out. Discard the token on the client.")
