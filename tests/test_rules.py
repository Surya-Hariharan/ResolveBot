from datetime import date

from app import rules

PARAMS = {"return_window_days": "30", "standard_warranty_months": "12", "extended_warranty_months": "24"}
TODAY = date(2026, 7, 1)


def item(returnable="yes", opened=0, name="Thing"):
    return {"item_id": "X-1", "name": name, "returnable": returnable, "opened": opened}


# --- return window: day 30 allowed, day 31 not (the example in the assignment)
def test_day_30_is_still_allowed():
    assert rules.check_return([item()], date(2026, 6, 1), "changed_mind", PARAMS, TODAY) is None


def test_day_31_is_too_late():
    code, _ = rules.check_return([item()], date(2026, 5, 31), "changed_mind", PARAMS, TODAY)
    assert code == "RETURN_WINDOW_EXPIRED"


def test_window_comes_from_parameters_not_code():
    delivered = date(2026, 6, 21)  # 10 days ago
    assert rules.check_return([item()], delivered, "changed_mind", PARAMS, TODAY) is None
    short = {**PARAMS, "return_window_days": "7"}
    code, _ = rules.check_return([item()], delivered, "changed_mind", short, TODAY)
    assert code == "RETURN_WINDOW_EXPIRED"
    # exactly on the boundary of the changed number
    assert rules.check_return([item()], date(2026, 6, 24), "changed_mind", short, TODAY) is None
    code, _ = rules.check_return([item()], date(2026, 6, 23), "changed_mind", short, TODAY)
    assert code == "RETURN_WINDOW_EXPIRED"


# --- the order of checks 2, 3, 4
def test_non_returnable_beats_expired_window():
    code, _ = rules.check_return([item("no")], date(2025, 1, 1), "changed_mind", PARAMS, TODAY)
    assert code == "NON_RETURNABLE"


def test_expired_window_beats_opened_earbuds():
    code, _ = rules.check_return(
        [item("unopened_only", opened=1)], date(2025, 1, 1), "changed_mind", PARAMS, TODAY
    )
    assert code == "RETURN_WINDOW_EXPIRED"


def test_opened_earbuds_changed_mind_is_refused():
    code, _ = rules.check_return(
        [item("unopened_only", opened=1)], date(2026, 6, 26), "changed_mind", PARAMS, TODAY
    )
    assert code == "OPENED_NOT_RETURNABLE"


def test_opened_earbuds_defective_is_eligible():
    assert (
        rules.check_return([item("unopened_only", opened=1)], date(2026, 6, 26), "defective", PARAMS, TODAY)
        is None
    )


def test_unopened_earbuds_changed_mind_is_eligible():
    assert (
        rules.check_return([item("unopened_only", opened=0)], date(2026, 6, 26), "changed_mind", PARAMS, TODAY)
        is None
    )


def test_gift_card_is_never_returnable_even_for_defects():
    code, _ = rules.check_return([item("no")], date(2026, 6, 29), "defective", PARAMS, TODAY)
    assert code == "NON_RETURNABLE"


def test_one_bad_item_blocks_the_whole_request():
    items = [item("yes"), item("no", name="Gift card")]
    code, message = rules.check_return(items, date(2026, 6, 29), "changed_mind", PARAMS, TODAY)
    assert code == "NON_RETURNABLE" and "Gift card" in message


# --- warranty arithmetic
def test_warranty_example_from_the_assignment():
    assert rules.warranty_end_date(date(2026, 3, 15), "standard", PARAMS) == date(2027, 3, 15)


def test_warranty_end_of_month_clamps_to_last_day():
    one_month = {**PARAMS, "standard_warranty_months": "1"}
    assert rules.warranty_end_date(date(2026, 1, 31), "standard", one_month) == date(2026, 2, 28)
    assert rules.warranty_end_date(date(2027, 1, 31), "standard", one_month) == date(2027, 2, 28)


def test_warranty_leap_year():
    one_month = {**PARAMS, "standard_warranty_months": "1"}
    assert rules.warranty_end_date(date(2028, 1, 31), "standard", one_month) == date(2028, 2, 29)
    assert rules.warranty_end_date(date(2024, 2, 29), "standard", PARAMS) == date(2025, 2, 28)
    assert rules.warranty_end_date(date(2024, 2, 29), "extended", PARAMS) == date(2026, 2, 28)


def test_warranty_end_date_itself_is_covered_next_day_is_not():
    delivered = date(2025, 7, 1)  # 12 months -> 1 July 2026 = today
    assert rules.warranty_is_active(delivered, "standard", PARAMS, TODAY)
    assert not rules.warranty_is_active(delivered, "standard", PARAMS, date(2026, 7, 2))


def test_extended_class_uses_the_extended_months():
    delivered = date(2024, 7, 1)
    assert rules.warranty_end_date(delivered, "extended", PARAMS) == date(2026, 7, 1)
    changed = {**PARAMS, "extended_warranty_months": "36"}
    assert rules.warranty_end_date(delivered, "extended", changed) == date(2027, 7, 1)


def test_class_none_has_no_warranty():
    assert rules.warranty_end_date(date(2026, 6, 1), "none", PARAMS) is None
    assert not rules.warranty_is_active(date(2026, 6, 1), "none", PARAMS, TODAY)


# --- warranty ticket and cancellation
def test_warranty_ticket_outcomes():
    live = {"warranty_class": "standard"}
    assert rules.check_warranty_ticket([live], date(2026, 6, 1), PARAMS, TODAY) is None
    assert rules.check_warranty_ticket([live], date(2024, 1, 1), PARAMS, TODAY)[0] == "WARRANTY_EXPIRED"
    assert rules.check_warranty_ticket([{"warranty_class": "none"}], date(2026, 6, 1), PARAMS, TODAY)[0] == "NO_WARRANTY"
    assert rules.check_warranty_ticket([live], None, PARAMS, TODAY)[0] == "NOT_DELIVERED"


def test_one_item_under_warranty_is_enough():
    items = [{"warranty_class": "none"}, {"warranty_class": "standard"}]
    assert rules.check_warranty_ticket(items, date(2026, 6, 1), PARAMS, TODAY) is None


def test_cancellation_only_while_processing():
    assert rules.check_cancellation("processing") is None
    assert rules.check_cancellation("shipped")[0] == "ORDER_ALREADY_SHIPPED"
    assert rules.check_cancellation("delivered")[0] == "ORDER_ALREADY_SHIPPED"
    assert rules.check_cancellation("cancelled")[0] == "ORDER_ALREADY_CANCELLED"


def test_display_date_has_no_leading_zero():
    assert rules.display_date(date(2026, 7, 3)) == "3 July 2026"
