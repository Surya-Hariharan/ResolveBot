import pytest

from app import tools
from app.tools import ToolContext
from tests.make_test_db import TODAY, count_rows, make_db

NOT_FOUND_TEXT = "I couldn't find that order on your account."


@pytest.fixture
def db(tmp_path):
    return make_db(tmp_path / "tools.db")


def ctx_for(db, customer="C1001"):
    return ToolContext(db_path=db, today=TODAY, customer_id=customer)


# ---------------------------------------------------------------- get_orders
def test_get_orders_computes_dates_and_warranty(db):
    result = tools.get_orders(ctx_for(db), order_id="SS-1")
    order = result["orders"][0]
    assert order["days_since_delivery"] == 10
    assert order["delivery_date"] == "21 June 2026"
    item = order["items"][0]
    assert item["warranty_end_date"] == "21 June 2027"
    assert item["warranty_active"] is True


def test_get_orders_marks_expired_warranty(db):
    order = tools.get_orders(ctx_for(db), order_id="SS-11")["orders"][0]
    assert order["items"][0]["warranty_active"] is False


def test_get_orders_undelivered_has_no_delivery_numbers(db):
    order = tools.get_orders(ctx_for(db), order_id="SS-7")["orders"][0]
    assert order["delivery_date"] is None and order["days_since_delivery"] is None
    assert order["shipment"]["expected_delivery_date"] == "4 July 2026"


def test_get_orders_lists_only_my_orders_and_filters_status(db):
    ids = {o["order_id"] for o in tools.get_orders(ctx_for(db))["orders"]}
    assert "SS-9" not in ids and "SS-1" in ids
    processing = tools.get_orders(ctx_for(db), status="processing")["orders"]
    assert [o["order_id"] for o in processing] == ["SS-6"]


def test_get_orders_bad_status_is_an_error_not_a_crash(db):
    assert tools.get_orders(ctx_for(db), status="lost")["error_code"] == "INVALID_INPUT"


def test_missing_and_foreign_orders_give_the_same_error(db):
    missing = tools.get_orders(ctx_for(db), order_id="SS-99999")
    foreign = tools.get_orders(ctx_for(db), order_id="SS-9")  # belongs to C1002
    assert missing == foreign
    assert missing["error_code"] == "ORDER_NOT_FOUND"
    assert missing["message"] == NOT_FOUND_TEXT


def test_order_ids_are_matched_ignoring_case_and_spaces(db):
    assert tools.get_orders(ctx_for(db), order_id=" ss-1 ")["count"] == 1


def test_sql_injection_text_is_just_an_unknown_order(db):
    result = tools.get_orders(ctx_for(db), order_id="SS-1' OR '1'='1")
    assert result["error_code"] == "ORDER_NOT_FOUND"


# ------------------------------------------------------ check_return_eligibility
def check(db, order, items, reason, customer="C1001"):
    return tools.check_return_eligibility(ctx_for(db, customer), order, items, reason)


def test_eligible_return(db):
    result = check(db, "SS-1", ["SS-1-1"], "changed_mind")
    assert result["eligible"] is True and result["days_since_delivery"] == 10


def test_day_30_eligible_day_31_expired(db):
    assert check(db, "SS-2", ["SS-2-1"], "changed_mind")["eligible"] is True
    assert check(db, "SS-3", ["SS-3-1"], "changed_mind")["error_code"] == "RETURN_WINDOW_EXPIRED"


def test_opened_earbuds_depend_on_reason(db):
    assert check(db, "SS-4", ["SS-4-1"], "changed_mind")["error_code"] == "OPENED_NOT_RETURNABLE"
    assert check(db, "SS-4", ["SS-4-1"], "defective")["eligible"] is True


def test_gift_card_and_software_are_non_returnable(db):
    assert check(db, "SS-5", ["SS-5-1"], "changed_mind")["error_code"] == "NON_RETURNABLE"
    assert check(db, "SS-12", ["SS-12-1"], "defective")["error_code"] == "NON_RETURNABLE"


def test_validation_errors(db):
    assert check(db, "SS-1", ["SS-1-9"], "changed_mind")["error_code"] == "INVALID_ITEM"
    assert check(db, "SS-1", ["SS-4-1"], "changed_mind")["error_code"] == "INVALID_ITEM"
    assert check(db, "SS-1", ["SS-1-1"], "bored")["error_code"] == "INVALID_REASON"
    assert check(db, "SS-6", ["SS-6-1"], "changed_mind")["error_code"] == "NOT_DELIVERED"
    assert check(db, "SS-1", [], "changed_mind")["error_code"] == "INVALID_INPUT"
    assert check(db, "SS-1", "SS-1-1", "changed_mind")["error_code"] == "INVALID_INPUT"


def test_invalid_item_message_matches_the_assignment_example(db):
    result = check(db, "SS-1", ["SS-1-9"], "changed_mind")
    assert result == {"error_code": "INVALID_ITEM", "message": "Item SS-1-9 is not in this order"}


def test_return_check_for_a_foreign_order_looks_like_a_missing_one(db):
    foreign = check(db, "SS-9", ["SS-9-1"], "changed_mind")
    missing = check(db, "SS-404", ["SS-404-1"], "changed_mind")
    assert foreign == missing and foreign["error_code"] == "ORDER_NOT_FOUND"


def test_order_check_comes_before_reason_check(db):
    # a bad reason must not reveal that a foreign order exists
    assert check(db, "SS-9", ["SS-9-1"], "bored")["error_code"] == "ORDER_NOT_FOUND"


def test_decisions_follow_the_database_parameters(tmp_path):
    strict = make_db(tmp_path / "strict.db", {**{"return_window_days": "7"},
                     "standard_warranty_months": "12", "extended_warranty_months": "24"})
    assert check(strict, "SS-1", ["SS-1-1"], "changed_mind")["error_code"] == "RETURN_WINDOW_EXPIRED"
    generous = make_db(tmp_path / "generous.db", {"return_window_days": "45",
                       "standard_warranty_months": "12", "extended_warranty_months": "24"})
    assert check(generous, "SS-3", ["SS-3-1"], "changed_mind")["eligible"] is True


# ------------------------------------------------------------- write tools
def test_create_return_request_saves_one_row(db):
    result = tools.create_return_request(ctx_for(db), "SS-1", ["SS-1-1"], "changed_mind")
    assert result["status"] == "requested"
    assert count_rows(db, "return_requests") == 1


def test_create_return_request_rechecks_and_saves_nothing_on_failure(db):
    result = tools.create_return_request(ctx_for(db), "SS-3", ["SS-3-1"], "changed_mind")
    assert result["error_code"] == "RETURN_WINDOW_EXPIRED"
    assert count_rows(db, "return_requests") == 0


def test_cannot_save_a_return_for_another_customers_order(db):
    result = tools.create_return_request(ctx_for(db), "SS-9", ["SS-9-1"], "changed_mind")
    assert result["error_code"] == "ORDER_NOT_FOUND"
    assert count_rows(db, "return_requests") == 0


def test_cancellation_ticket_only_while_processing(db):
    ok = tools.create_support_ticket(ctx_for(db), "SS-6", "cancellation", "Please cancel")
    assert ok["status"] == "open" and count_rows(db, "support_tickets") == 1
    shipped = tools.create_support_ticket(ctx_for(db), "SS-7", "cancellation", "Please cancel")
    assert shipped["error_code"] == "ORDER_ALREADY_SHIPPED"
    assert count_rows(db, "support_tickets") == 1


def test_warranty_ticket_rules(db):
    ok = tools.create_support_ticket(ctx_for(db), "SS-1", "warranty", "Stopped working")
    assert "ticket_id" in ok
    expired = tools.create_support_ticket(ctx_for(db), "SS-11", "warranty", "Broken")
    assert expired["error_code"] == "WARRANTY_EXPIRED"
    none = tools.create_support_ticket(ctx_for(db), "SS-5", "warranty", "Broken")
    assert none["error_code"] == "NO_WARRANTY"
    not_delivered = tools.create_support_ticket(ctx_for(db), "SS-7", "warranty", "Broken")
    assert not_delivered["error_code"] == "NOT_DELIVERED"
    assert count_rows(db, "support_tickets") == 1


def test_ticket_input_errors(db):
    assert tools.create_support_ticket(ctx_for(db), "SS-6", "damage", "x")["error_code"] == "INVALID_TICKET_TYPE"
    assert tools.create_support_ticket(ctx_for(db), "SS-6", "cancellation", "  ")["error_code"] == "INVALID_INPUT"
    assert tools.create_support_ticket(ctx_for(db), "SS-9", "cancellation", "x")["error_code"] == "ORDER_NOT_FOUND"


def test_validate_write_never_saves(db):
    assert tools.validate_write(ctx_for(db), "create_return_request",
                                {"order_id": "SS-1", "item_ids": ["SS-1-1"], "reason": "changed_mind"}) is None
    assert tools.validate_write(ctx_for(db), "create_support_ticket",
                                {"order_id": "SS-6", "ticket_type": "cancellation", "summary": "x"}) is None
    assert count_rows(db, "return_requests") == 0 and count_rows(db, "support_tickets") == 0


def test_read_tools_do_not_write(db):
    tools.get_orders(ctx_for(db))
    check(db, "SS-1", ["SS-1-1"], "changed_mind")
    assert count_rows(db, "return_requests") == 0 and count_rows(db, "support_tickets") == 0


def test_exactly_five_tools_are_registered_with_the_assignment_names():
    names = [schema["function"]["name"] for schema in tools.TOOL_SCHEMAS]
    assert names == ["get_orders", "search_policy", "check_return_eligibility",
                     "create_return_request", "create_support_ticket"]
    assert set(tools.TOOL_FUNCTIONS) == set(names)


def test_no_tool_accepts_a_customer_id():
    for name, allowed in tools.TOOL_ARGUMENTS.items():
        assert "customer_id" not in allowed, name


def test_search_policy_validates_input_and_missing_index(db):
    assert tools.search_policy(ctx_for(db), "")["error_code"] == "INVALID_INPUT"
    assert tools.search_policy(ctx_for(db), "returns")["error_code"] == "POLICY_UNAVAILABLE"
