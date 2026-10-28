"""The five tools the assistant can use.

Every tool gets the customer id from `ToolContext`, which the API fills in from
the X-Customer-Id header. The LLM never supplies it. Every order lookup uses
"WHERE order_id = ? AND customer_id = ?", so another customer's order looks
exactly like an order that does not exist.
"""
import json
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime

from app import database, rules

ORDER_NOT_FOUND_TEXT = "I couldn't find that order on your account."
ORDER_STATUSES = ("processing", "shipped", "delivered", "cancelled")
MAX_ORDERS_RETURNED = 20


@dataclass
class ToolContext:
    db_path: str
    today: date
    customer_id: str  # from the header, never from the LLM
    policy_search: object = None  # callable(query) -> list of paragraphs


def error(code, message):
    return {"error_code": code, "message": message}


def clean_id(value):
    """Order and item ids are matched case-insensitively and without spaces."""
    return value.strip().upper() if isinstance(value, str) else None


def delivery_date_of(order, shipment):
    """The date the order was delivered, or None if it has not been (as of today)."""
    if order["status"] != "delivered" or shipment is None or not shipment["delivered_at"]:
        return None
    return datetime.fromisoformat(shipment["delivered_at"]).date()


def find_order(connection, customer_id, order_id):
    """Return (order, shipment, items) or None. Foreign orders return None too."""
    order = connection.execute(
        "SELECT * FROM orders WHERE order_id = ? AND customer_id = ?", (order_id, customer_id)
    ).fetchone()
    if order is None:
        return None
    shipment = connection.execute(
        "SELECT * FROM shipments WHERE order_id = ?", (order_id,)
    ).fetchone()
    items = connection.execute(
        """SELECT i.item_id, i.qty, i.unit_price, i.opened, i.defect_reported,
                  p.name, p.category, p.returnable, p.warranty_class
           FROM order_items i JOIN products p ON p.product_id = i.product_id
           WHERE i.order_id = ? ORDER BY i.item_id""",
        (order_id,),
    ).fetchall()
    return order, shipment, items


def describe_order(order, shipment, items, params, today):
    """The order as the LLM sees it, including the values we work out in Python."""
    delivered_on = delivery_date_of(order, shipment)
    result = {
        "order_id": order["order_id"],
        "order_date": rules.display_date(datetime.fromisoformat(order["order_date"]).date()),
        "status": order["status"],
        "total_inr": order["total"],
        "delivery_date": rules.display_date(delivered_on) if delivered_on else None,
        "days_since_delivery": rules.days_since(delivered_on, today) if delivered_on else None,
        "shipment": None,
        "items": [],
    }
    if shipment is not None:
        result["shipment"] = {
            "carrier": shipment["carrier"],
            "status": shipment["status"],
            "expected_delivery_date": _display_iso(shipment["eta"]),
            "original_expected_delivery_date": _display_iso(shipment["original_eta"]),
        }
    for item in items:
        warranty_end = None
        warranty_active = False
        if delivered_on:
            end = rules.warranty_end_date(delivered_on, item["warranty_class"], params)
            if end:
                warranty_end = rules.display_date(end)
                warranty_active = today <= end
        result["items"].append(
            {
                "item_id": item["item_id"],
                "name": item["name"],
                "category": item["category"],
                "qty": item["qty"],
                "opened": bool(item["opened"]),
                "returnable": item["returnable"],
                "warranty_class": item["warranty_class"],
                "warranty_end_date": warranty_end,
                "warranty_active": warranty_active,
            }
        )
    return result


def _display_iso(text):
    return rules.display_date(date.fromisoformat(text[:10])) if text else None


# ---------------------------------------------------------------- read tools


def get_orders(ctx, order_id=None, status=None):
    if status is not None and status not in ORDER_STATUSES:
        return error("INVALID_INPUT", f"status must be one of {', '.join(ORDER_STATUSES)}.")
    params = database.load_policy_parameters(ctx.db_path)
    with closing(database.connect(ctx.db_path)) as connection:
        if order_id is not None:
            found = find_order(connection, ctx.customer_id, clean_id(order_id) or "")
            if found is None:
                return error("ORDER_NOT_FOUND", ORDER_NOT_FOUND_TEXT)
            order, shipment, items = found
            if status is not None and order["status"] != status:
                return {"orders": [], "count": 0}
            return {"orders": [describe_order(order, shipment, items, params, ctx.today)], "count": 1}

        query = "SELECT order_id FROM orders WHERE customer_id = ?"
        values = [ctx.customer_id]
        if status is not None:
            query += " AND status = ?"
            values.append(status)
        query += " ORDER BY order_date DESC"
        order_ids = [row["order_id"] for row in connection.execute(query, values).fetchall()]
        orders = []
        for one_id in order_ids[:MAX_ORDERS_RETURNED]:
            order, shipment, items = find_order(connection, ctx.customer_id, one_id)
            orders.append(describe_order(order, shipment, items, params, ctx.today))
    result = {"orders": orders, "count": len(order_ids)}
    if len(order_ids) > MAX_ORDERS_RETURNED:
        result["note"] = f"Only the {MAX_ORDERS_RETURNED} most recent orders are shown."
    return result


def search_policy(ctx, query=None):
    if not isinstance(query, str) or not query.strip():
        return error("INVALID_INPUT", "query must be a non-empty string.")
    if ctx.policy_search is None:
        return error("POLICY_UNAVAILABLE", "The policy documents could not be loaded.")
    return {"paragraphs": ctx.policy_search(query.strip())}


def _check_return(connection, ctx, order_id, item_ids, reason):
    """Shared by the eligibility tool and the return request. Returns (error, details)."""
    if not isinstance(order_id, str) or not order_id.strip():
        return error("INVALID_INPUT", "order_id is required."), None
    found = find_order(connection, ctx.customer_id, clean_id(order_id))
    if found is None:
        return error("ORDER_NOT_FOUND", ORDER_NOT_FOUND_TEXT), None
    order, shipment, order_items = found

    if reason not in rules.RETURN_REASONS:
        return error("INVALID_REASON", "reason must be changed_mind or defective."), None

    wanted = [clean_id(i) for i in item_ids] if isinstance(item_ids, list) else []
    if not wanted or None in wanted:
        return error("INVALID_INPUT", "item_ids must be a non-empty list of item ids."), None
    by_id = {item["item_id"]: item for item in order_items}
    for item_id in wanted:
        if item_id not in by_id:
            return error("INVALID_ITEM", f"Item {item_id} is not in this order"), None

    delivered_on = delivery_date_of(order, shipment)
    if delivered_on is None or delivered_on > ctx.today:
        return error("NOT_DELIVERED", "This order has not been delivered yet."), None

    params = database.load_policy_parameters(ctx.db_path)
    chosen = [dict(by_id[i]) for i in dict.fromkeys(wanted)]
    problem = rules.check_return(chosen, delivered_on, reason, params, ctx.today)
    if problem:
        return error(*problem), None
    details = {
        "order_id": order["order_id"],
        "item_ids": [i["item_id"] for i in chosen],
        "item_names": [i["name"] for i in chosen],
        "reason": reason,
        "days_since_delivery": rules.days_since(delivered_on, ctx.today),
        "return_window_days": int(params["return_window_days"]),
    }
    return None, details


def check_return_eligibility(ctx, order_id=None, item_ids=None, reason=None):
    with closing(database.connect(ctx.db_path)) as connection:
        problem, details = _check_return(connection, ctx, order_id, item_ids, reason)
    if problem:
        return problem
    return {"eligible": True, **details}


def _check_ticket(connection, ctx, order_id, ticket_type, summary):
    """Shared by the proposal check and the real ticket. Returns (error, details)."""
    if not isinstance(order_id, str) or not order_id.strip():
        return error("INVALID_INPUT", "order_id is required."), None
    found = find_order(connection, ctx.customer_id, clean_id(order_id))
    if found is None:
        return error("ORDER_NOT_FOUND", ORDER_NOT_FOUND_TEXT), None
    order, shipment, items = found

    if ticket_type not in rules.TICKET_TYPES:
        return error("INVALID_TICKET_TYPE", "ticket_type must be warranty or cancellation."), None
    if not isinstance(summary, str) or not summary.strip():
        return error("INVALID_INPUT", "summary must be a short non-empty description."), None

    if ticket_type == "cancellation":
        problem = rules.check_cancellation(order["status"])
    else:
        params = database.load_policy_parameters(ctx.db_path)
        problem = rules.check_warranty_ticket(
            [dict(i) for i in items], delivery_date_of(order, shipment), params, ctx.today
        )
    if problem:
        return error(*problem), None
    return None, {"order_id": order["order_id"], "ticket_type": ticket_type, "summary": summary.strip()}


# --------------------------------------------------------------- write tools


def validate_write(ctx, name, args):
    """Check a write without saving. Used when the assistant only proposes it."""
    with closing(database.connect(ctx.db_path)) as connection:
        if name == "create_return_request":
            problem, _ = _check_return(
                connection, ctx, args.get("order_id"), args.get("item_ids"), args.get("reason")
            )
        else:
            problem, _ = _check_ticket(
                connection, ctx, args.get("order_id"), args.get("ticket_type"), args.get("summary")
            )
    return problem


def create_return_request(ctx, order_id=None, item_ids=None, reason=None):
    """Re-check, then save, inside one transaction so a failed check saves nothing."""
    now = datetime.now().isoformat(timespec="seconds")
    with closing(database.connect(ctx.db_path)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        problem, details = _check_return(connection, ctx, order_id, item_ids, reason)
        if problem:
            connection.rollback()
            return problem
        request_id = str(uuid.uuid4())
        connection.execute(
            "INSERT INTO return_requests (request_id, order_id, customer_id, item_ids, reason, status, created_at)"
            " VALUES (?, ?, ?, ?, ?, 'requested', ?)",
            (request_id, details["order_id"], ctx.customer_id, json.dumps(details["item_ids"]), reason, now),
        )
        connection.commit()
    return {"request_id": request_id, "status": "requested", **details}


def create_support_ticket(ctx, order_id=None, ticket_type=None, summary=None):
    now = datetime.now().isoformat(timespec="seconds")
    with closing(database.connect(ctx.db_path)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        problem, details = _check_ticket(connection, ctx, order_id, ticket_type, summary)
        if problem:
            connection.rollback()
            return problem
        ticket_id = str(uuid.uuid4())
        connection.execute(
            "INSERT INTO support_tickets (ticket_id, order_id, customer_id, ticket_type, summary, status, created_at)"
            " VALUES (?, ?, ?, ?, ?, 'open', ?)",
            (ticket_id, details["order_id"], ctx.customer_id, ticket_type, details["summary"], now),
        )
        connection.commit()
    return {"ticket_id": ticket_id, "status": "open", **details}


# The only tools the LLM may call. Anything else is rejected by the orchestrator.
TOOL_FUNCTIONS = {
    "get_orders": get_orders,
    "search_policy": search_policy,
    "check_return_eligibility": check_return_eligibility,
    "create_return_request": create_return_request,
    "create_support_ticket": create_support_ticket,
}
WRITE_TOOLS = ("create_return_request", "create_support_ticket")


def _tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


ORDER_ID_PROPERTY = {"type": "string", "description": "Order id such as SS-10458."}
ITEM_IDS_PROPERTY = {
    "type": "array",
    "items": {"type": "string"},
    "description": "Item ids from get_orders, such as SS-10458-1. Never guess them.",
}
REASON_PROPERTY = {
    "type": "string",
    "enum": list(rules.RETURN_REASONS),
    "description": "changed_mind = the customer just does not want it. "
    "defective = broken, damaged or the wrong item was sent.",
}

TOOL_SCHEMAS = [
    _tool(
        "get_orders",
        "Look up the logged-in customer's orders with items, shipment status, delivery date, "
        "days since delivery, and each item's warranty end date and whether it is still active. "
        "Use it to track an order, to find which order or item the customer means, and for "
        "warranty questions. Give order_id for one order, or leave it out to list orders.",
        {
            "order_id": {**ORDER_ID_PROPERTY, "description": "Optional. One order id."},
            "status": {"type": "string", "enum": list(ORDER_STATUSES), "description": "Optional filter."},
        },
        [],
    ),
    _tool(
        "search_policy",
        "Search ShopSphere's policy documents and FAQ. Use it for any question about rules "
        "(returns, refunds, warranty, shipping, cancellation, exchanges, damage). Returns the "
        "best matching paragraphs with their doc_id and section.",
        {"query": {"type": "string", "description": "A full question in plain words, for example "
                   "'How many days do I have to return an item?'. Not single keywords."}},
        ["query"],
    ),
    _tool(
        "check_return_eligibility",
        "Check whether items in an order can be returned. Read-only. Returns eligible or an error code. "
        "Use it before proposing a return and whenever the customer asks if they can return something.",
        {"order_id": ORDER_ID_PROPERTY, "item_ids": ITEM_IDS_PROPERTY, "reason": REASON_PROPERTY},
        ["order_id", "item_ids", "reason"],
    ),
    _tool(
        "create_return_request",
        "Create a return request. Call it ONLY when the customer has asked for the return to be made. "
        "It does not save immediately: the customer must say yes first, so you will ask them to confirm.",
        {"order_id": ORDER_ID_PROPERTY, "item_ids": ITEM_IDS_PROPERTY, "reason": REASON_PROPERTY},
        ["order_id", "item_ids", "reason"],
    ),
    _tool(
        "create_support_ticket",
        "Open a support ticket: warranty (a faulty item under warranty) or cancellation (cancel an "
        "order that is still processing). Call it ONLY when the customer has asked for the ticket. "
        "It does not save immediately: the customer must say yes first.",
        {
            "order_id": ORDER_ID_PROPERTY,
            "ticket_type": {"type": "string", "enum": list(rules.TICKET_TYPES)},
            "summary": {"type": "string", "description": "One or two sentences describing the problem."},
        },
        ["order_id", "ticket_type", "summary"],
    ),
]
# Allowed argument names per tool; anything else the LLM sends (like a customer id) is dropped
TOOL_ARGUMENTS = {
    schema["function"]["name"]: list(schema["function"]["parameters"]["properties"])
    for schema in TOOL_SCHEMAS
}
