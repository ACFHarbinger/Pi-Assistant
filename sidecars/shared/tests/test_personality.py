from pi_sidecar.personality import Personality


def test_personality_default(tmp_path):
    """Test personality when no soul.md exists."""
    p = Personality(workspace_path=str(tmp_path))
    assert p.name == "Pi"
    assert "You are Pi" in p.system_prompt
    assert "Hey! I'm Pi" in p.hatching_message


def test_personality_load_soul(tmp_path):
    """Test loading personality from soul.md."""
    soul_file = tmp_path / "soul.md"
    soul_file.write_text("You are **Antigravity**\n\n# Hatching\n> Hello world")

    p = Personality(workspace_path=str(tmp_path))
    assert p.name == "Antigravity"
    assert "Hello world" in p.hatching_message
    assert "Personality Guide" in p.system_prompt


def test_personality_update_name(tmp_path):
    """Test updating name in soul.md."""
    soul_file = tmp_path / "soul.md"
    soul_file.write_text("You are **OldName**")

    p = Personality(workspace_path=str(tmp_path))
    assert p.name == "OldName"

    success = p.update_name("NewName")
    assert success is True
    assert p.name == "NewName"
    assert "You are **NewName**" in soul_file.read_text()


def test_personality_reload(tmp_path):
    """Test reloading soul.md."""
    soul_file = tmp_path / "soul.md"
    soul_file.write_text("You are **Version1**")

    p = Personality(workspace_path=str(tmp_path))
    assert p.name == "Version1"

    soul_file.write_text("You are **Version2**")
    p.reload()
    assert p.name == "Version2"
