"""Business rules: pure Python, no database and no LLM.

Every number comes from the `params` dict (the policy_parameters table), so a
different database with different numbers changes the decisions automatically.
Each check returns None when it passes, or an (error_code, message) pair.
"""
from dateutil.relativedelta import relativedelta

RETURN_REASONS = ("changed_mind", "defective")
TICKET_TYPES = ("warranty", "cancellation")


def display_date(day):
    """3 July 2026 (no leading zero, same style on every platform)."""
    return f"{day.day} {day.strftime('%B %Y')}"


def days_since(delivery_date, today):
    return (today - delivery_date).days


def warranty_months(warranty_class, params):
    """Months of warranty for a product class, or None when it has no warranty."""
    if warranty_class == "standard":
        return int(params["standard_warranty_months"])
    if warranty_class == "extended":
        return int(params["extended_warranty_months"])
    return None


def warranty_end_date(delivery_date, warranty_class, params):
    """Same day N months later; relativedelta clamps to the month's last day."""
    months = warranty_months(warranty_class, params)
    if months is None:
        return None
    return delivery_date + relativedelta(months=months)


def warranty_is_active(delivery_date, warranty_class, params, today):
    """The end date itself is still covered, so we compare with <=."""
    end = warranty_end_date(delivery_date, warranty_class, params)
    return end is not None and today <= end


def check_return(items, delivery_date, reason, params, today):
    """The return checks 2-4 from the assignment, in order.

    `items` are dicts with item_id, returnable ('yes' | 'unopened_only' | 'no'),
    opened (0/1) and name. Checks 1 (order, items, delivered) happen earlier,
    where the database is read.
    """
    # Check 2: some products can never be returned
    for item in items:
        if item["returnable"] == "no":
            return ("NON_RETURNABLE", f"{item['name']} cannot be returned.")

    # Check 3: the return window (day 30 is still allowed)
    window = int(params["return_window_days"])
    days = days_since(delivery_date, today)
    if days > window:
        return (
            "RETURN_WINDOW_EXPIRED",
            f"The order was delivered {days} days ago; the return window is {window} days.",
        )

    # Check 4: opened items can't be returned for a change of mind if sealed-only
    if reason == "changed_mind":
        for item in items:
            if item["opened"] and item["returnable"] == "unopened_only":
                return (
                    "OPENED_NOT_RETURNABLE",
                    f"{item['name']} has been opened and can only be returned unopened "
                    "when you changed your mind.",
                )
    return None


def check_warranty_ticket(items, delivery_date, params, today):
    """Warranty ticket: delivered, and at least one item still under warranty."""
    if delivery_date is None or delivery_date > today:
        return ("NOT_DELIVERED", "This order has not been delivered yet.")
    has_warranty = [i for i in items if warranty_months(i["warranty_class"], params)]
    if not has_warranty:
        return ("NO_WARRANTY", "None of the items in this order have a warranty.")
    if not any(warranty_is_active(delivery_date, i["warranty_class"], params, today) for i in has_warranty):
        return ("WARRANTY_EXPIRED", "The warranty has ended for every item in this order.")
    return None


def check_cancellation(order_status):
    """Only an order that is still processing can be cancelled."""
    if order_status == "processing":
        return None
    if order_status == "cancelled":
        return ("ORDER_ALREADY_CANCELLED", "This order has already been cancelled.")
    return ("ORDER_ALREADY_SHIPPED", "This order has already shipped, so it can't be cancelled.")
