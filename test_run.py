import os
import sys
import uuid
import time

# Add local directory to path to import without installing
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from execlave import Execlave, AgentPausedError


def test_sdk():
    api_key = os.environ.get("TEST_API_KEY", "exe_dev_mock_key")
    print(f"Initializing Execlave SDK with API key: {api_key[:10]}...")

    exe = Execlave(
        api_key=api_key,
        environment="development",
        async_mode=False,   # sync for test visibility
        debug=True,
        batch_size=10,
    )

    # 1. Connectivity
    print("\n1. Testing connectivity (/api/health)...")
    if exe.ping():
        print("   ✅ Connected to Execlave API")
    else:
        print("   ❌ Could not connect. Is the backend running?")
        return

    # 2. Register agent
    print("\n2. Testing register_agent()...")
    test_id = f"sdk-test-{uuid.uuid4().hex[:6]}"
    try:
        agent = exe.register_agent(
            agent_id=test_id,
            name="SDK Integration Test Agent",
            type="autonomous",
            platform="custom",
            environment="development",
            description="Agent created via Python SDK test",
            tags=["test", "sdk"],
            metadata={"test_run": "true"},
        )
        print(f"   ✅ Registered agent: {agent.name} (id={agent.id})")
    except Exception as e:
        print(f"   ❌ Failed to register agent: {e}")
        return

    # 3. Test @exe.trace decorator
    print("\n3. Testing @exe.trace decorator...")

    @exe.trace
    def answer_question(question: str) -> str:
        time.sleep(0.01)  # Simulate work
        return f"Answer to: {question}"

    try:
        result = answer_question("What is Execlave?")
        print(f"   ✅ Decorator trace completed: {result}")
    except Exception as e:
        print(f"   ❌ Decorator trace failed: {e}")

    # 4. Test context manager trace
    print("\n4. Testing context manager trace...")
    try:
        with exe.trace(session_id="test-session", user_id="test-user") as t:
            t.set_input("Hello, world!")
            time.sleep(0.01)
            t.set_output("Hi there!")
            t.set_model("gpt-4-turbo")
            t.set_tokens(input=5, output=3)
            t.set_cost(0.001)
            t.add_metadata({"confidence": 0.95})
        print("   ✅ Context manager trace completed")
    except Exception as e:
        print(f"   ❌ Context manager trace failed: {e}")

    # 5. Test manual trace
    print("\n5. Testing start_trace() manual trace...")
    try:
        trace = exe.start_trace(trace_id="manual-test-001", session_id="test-session")
        trace.set_input("Manual input")
        time.sleep(0.01)
        trace.set_output("Manual output")
        trace.finish(status="success")
        print("   ✅ Manual trace completed")
    except Exception as e:
        print(f"   ❌ Manual trace failed: {e}")

    # 6. Flush
    print("\n6. Testing flush()...")
    try:
        exe.flush()
        print("   ✅ Flush completed")
    except Exception as e:
        print(f"   ❌ Flush failed: {e}")

    # 7. Shutdown
    print("\n7. Testing shutdown()...")
    exe.shutdown()
    print("   ✅ SDK shutdown completed")

    print("\n✅ All SDK tests completed!")


if __name__ == "__main__":
    test_sdk()
