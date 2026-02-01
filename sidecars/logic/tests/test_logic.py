import pytest
from pi_sidecar.logic_main import LogicRequestHandler


@pytest.mark.asyncio
async def test_logic_handler_ping():
    """Test health ping returns status ok."""
    handler = LogicRequestHandler()
    result = await handler.dispatch("health.ping", {}, None)
    assert result["status"] == "ok"
    assert result["sidecar"] == "logic"


@pytest.mark.asyncio
async def test_logic_handler_unknown_method():
    """Test unknown method raises ValueError."""
    handler = LogicRequestHandler()
    with pytest.raises(ValueError, match="not supported"):
        await handler.dispatch("unknown.method", {}, None)


@pytest.mark.asyncio
async def test_logic_handler_personality_name(tmp_path, monkeypatch):
    """Test personality name retrieval and update."""
    from pi_sidecar.personality import Personality

    # Mock soul.md
    soul_file = tmp_path / "soul.md"
    soul_file.write_text("You are **TestAgent**")

    # Create personality with tmp_path
    p = Personality(workspace_path=str(tmp_path))

    # Mock get_personality to return our instance
    monkeypatch.setattr("pi_sidecar.personality.get_personality", lambda: p)

    handler = LogicRequestHandler()

    # Test get_name
    result = await handler.dispatch("personality.get_name", {}, None)
    assert result["name"] == "TestAgent"

    # Test update_name
    result = await handler.dispatch("personality.update_name", {"name": "NewAgent"}, None)
    assert result["success"] is True
    assert result["name"] == "NewAgent"
    assert "You are **NewAgent**" in soul_file.read_text()


@pytest.mark.asyncio
async def test_logic_handler_personality_hatching(tmp_path, monkeypatch):
    """Test hatching message retrieval."""
    from pi_sidecar.personality import Personality

    soul_file = tmp_path / "soul.md"
    soul_file.write_text("You are **Pi**\n\n# First Encounter\n> Welcome to Pi!")

    p = Personality(workspace_path=str(tmp_path))
    monkeypatch.setattr("pi_sidecar.personality.get_personality", lambda: p)

    handler = LogicRequestHandler()
    result = await handler.dispatch("personality.get_hatching", {}, None)
    assert "Welcome to Pi!" in result["message"]


@pytest.mark.asyncio
async def test_logic_handler_personality_prompt(tmp_path, monkeypatch):
    """Test system prompt retrieval."""
    from pi_sidecar.personality import Personality

    soul_file = tmp_path / "soul.md"
    soul_file.write_text("You are **Pi**\n\nBe helpful and friendly.")

    p = Personality(workspace_path=str(tmp_path))
    monkeypatch.setattr("pi_sidecar.personality.get_personality", lambda: p)

    handler = LogicRequestHandler()
    result = await handler.dispatch("personality.get_prompt", {}, None)
    assert "Personality Guide" in result["prompt"]
    assert "Be helpful and friendly" in result["prompt"]


@pytest.mark.asyncio
async def test_logic_handler_all_registered_methods():
    """Test that all expected handlers are registered."""
    handler = LogicRequestHandler()
    expected_methods = [
        "health.ping",
        "lifecycle.shutdown",
        "personality.get_hatching",
        "personality.get_prompt",
        "personality.get_name",
        "personality.update_name",
    ]
    for method in expected_methods:
        assert method in handler._handlers, f"Missing handler for {method}"
