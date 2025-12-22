import sys
import os

# Add paths
sys.path.append(os.path.abspath("packages/api/src"))
sys.path.append(os.path.abspath("packages/rag/src"))

try:
    from omniscient_api.app import create_app
    print("Import create_app successful")
    from omniscient_api.auth import verify_api_key
    print("Import verify_api_key successful")
    import tests.test_auth
    print("Import tests.test_auth successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
