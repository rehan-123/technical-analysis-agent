from __future__ import annotations

from portfolio.portfolio_models import CashBalance
from portfolio.portfolio_validation import InsufficientFundsError


class CashManager:
    """Pure cash arithmetic over an immutable ``CashBalance``.

    Every operation returns a new balance rather than mutating one, so cash
    changes are explicit at the call site. Buying power is modelled as settled
    cash minus reservations — no leverage or margin, which is broker-specific
    and deliberately out of scope.
    """

    @staticmethod
    def buying_power(cash: CashBalance) -> float:
        """Cash currently deployable."""
        return cash.available

    @staticmethod
    def can_afford(cash: CashBalance, amount: float) -> bool:
        """Whether ``amount`` fits inside available cash."""
        if amount < 0:
            raise InsufficientFundsError("amount must not be negative")
        return amount <= cash.available

    @classmethod
    def debit(cls, cash: CashBalance, amount: float) -> CashBalance:
        """Spend ``amount``.

        Raises:
            InsufficientFundsError: if it exceeds available cash — the domain
                forbids negative balances outright.
        """
        if not cls.can_afford(cash, amount):
            raise InsufficientFundsError(
                f"insufficient funds: need {amount:.2f}, available {cash.available:.2f}"
            )
        return cash.model_copy(update={"amount": round(cash.amount - amount, 2)})

    @staticmethod
    def credit(cash: CashBalance, amount: float) -> CashBalance:
        """Add settled cash (a sale, deposit, or dividend)."""
        if amount < 0:
            raise InsufficientFundsError("credit amount must not be negative")
        return cash.model_copy(update={"amount": round(cash.amount + amount, 2)})

    @classmethod
    def reserve(cls, cash: CashBalance, amount: float) -> CashBalance:
        """Commit cash to a pending order without removing it from the balance."""
        if not cls.can_afford(cash, amount):
            raise InsufficientFundsError(
                f"cannot reserve {amount:.2f}; available {cash.available:.2f}"
            )
        return cash.model_copy(update={"reserved": round(cash.reserved + amount, 2)})

    @staticmethod
    def release(cash: CashBalance, amount: float) -> CashBalance:
        """Release a reservation, clamped so reserved can never go negative."""
        if amount < 0:
            raise InsufficientFundsError("release amount must not be negative")
        return cash.model_copy(update={"reserved": round(max(0.0, cash.reserved - amount), 2)})
