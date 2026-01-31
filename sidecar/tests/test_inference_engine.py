import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock
from pi_sidecar.inference.engine import InferenceEngine



@pytest.mark.asyncio
async def test_plan_generation_prompt(engine, mock_registry):
    """Verify that the planning prompt is constructed correctly."""
    mock_registry.load_model.return_value = None 
    
    # Use explicit AsyncMock assigned to the instance method
    mock_response = {"text": "{\"tool_calls\": [], \"reasoning\": \"test\"}"}
    engine.complete = AsyncMock(return_value=mock_response)

    task = "List files"
    iteration = 1
    context = [{"role": "user", "content": "previous message"}]

    plan = await engine.plan(task, iteration, context)

    # Verify complete was called
    engine.complete.assert_called_once()
    call_kwargs = engine.complete.call_args[1]
    prompt = call_kwargs["prompt"]
    
    # Check that key components are in the prompt
    assert f"Task: {task}" in prompt
    assert f"Iteration: {iteration}" in prompt
    assert "previous message" in prompt
    assert "You are an AI agent planner" in prompt

@pytest.mark.asyncio
async def test_embed_uses_sentence_transformer(engine):
    """Verify embed uses the cached model."""
    mock_model = MagicMock()
    # Mock the numpy array returned by encode
    mock_numpy_array = MagicMock()
    mock_numpy_array.tolist.return_value = [0.1, 0.2, 0.3]
    mock_model.encode.return_value = mock_numpy_array
    
    # Manually inject the mock model to bypass local loading
    engine._embedding_model = mock_model

    vector = await engine.embed("hello world", "test-model")

    mock_model.encode.assert_called_with("hello world", convert_to_numpy=True)
    assert vector == [0.1, 0.2, 0.3]
