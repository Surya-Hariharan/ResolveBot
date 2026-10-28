from tests.conftest import chat

GOOD_BODY = {"session_id": "s1", "message": "hello"}


def test_reply_has_every_required_field(client):
    response = chat(client, GOOD_BODY)
    assert response.status_code == 200
    reply = response.json()
    assert set(reply) == {
        "request_id", "answer", "route", "citations", "tool_calls", "pending_confirmation",
    }
    assert reply["route"] in {"answer", "clarify", "no_answer", "out_of_scope"}
    assert reply["pending_confirmation"] is False


def test_no_api_key_gives_a_controlled_reply_not_a_crash(client):
    reply = chat(client, GOOD_BODY).json()
    assert reply["route"] == "no_answer"
    assert "API key" in reply["answer"]


def test_missing_customer_header_is_401(client):
    assert chat(client, GOOD_BODY, customer=None).status_code == 401


def test_unknown_customer_is_401(client):
    assert chat(client, GOOD_BODY, customer="C9999").status_code == 401


def test_invalid_json_is_400(client):
    response = client.post(
        "/chat", content="{not json", headers={"X-Customer-Id": "C1001"}
    )
    assert response.status_code == 400


def test_missing_fields_are_400(client):
    assert chat(client, {"message": "hi"}).status_code == 400
    assert chat(client, {"session_id": "s1"}).status_code == 400
    assert chat(client, {"session_id": "s1", "message": "  "}).status_code == 400


def test_wrong_types_are_400(client):
    assert chat(client, {"session_id": 5, "message": "hi"}).status_code == 400
    assert chat(client, ["not", "an", "object"]).status_code == 400


def test_unauthenticated_beats_bad_body(client):
    assert chat(client, {}, customer="C9999").status_code == 401


def test_other_customer_cannot_use_my_session(client):
    assert chat(client, GOOD_BODY, customer="C1001").status_code == 200
    assert chat(client, GOOD_BODY, customer="C1002").status_code == 403
    # and the owner can still keep using it
    assert chat(client, GOOD_BODY, customer="C1001").status_code == 200
