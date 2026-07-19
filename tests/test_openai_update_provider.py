from surrogate_rollout.optimization.policies import openai_update


def test_openai_update_provider_uses_strict_json_schema(monkeypatch):
    captured = {}

    def fake_chat(prompt, **kwargs):
        captured.update(kwargs)
        return '{"schema_version":"memory_codebook_updater_response_v2","actions":[]}'

    monkeypatch.setattr(openai_update, "_openai_chat", fake_chat)
    schema = openai_update.codebook_updater_response_schema(max_actions=7)
    provider = openai_update.OpenAIJSONUpdateProvider(
        model="fixture", api_key="fixture", max_calls=3,
        response_schema_name="phase4_codebook_update_plan",
        response_schema=schema)
    provider("system", "request")
    response_format = captured["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == schema
    assert schema["properties"]["actions"]["maxItems"] == 7
    assert set(schema["required"]) == {"schema_version", "actions"}
    assert "proposed_property_id" not in schema["properties"]["actions"][
        "items"]["properties"]


def test_router_schema_requires_target_and_envelope():
    schema = openai_update.router_updater_response_schema(max_actions=5)
    action_schema = schema["properties"]["actions"]["items"]
    assert "target_property_id" in action_schema["required"]
    assert schema["properties"]["schema_version"]["enum"] == [
        "memory_router_updater_response_v1"]
