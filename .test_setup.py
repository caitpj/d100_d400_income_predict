import sys
import time

print("------------------------------------------------")
print("1. STARTING SYSTEM CHECK...")
print(f"   Python Version: {sys.version.split()[0]}")

try:
    print("2. IMPORTING LIBRARY...")
    from ucimlrepo import fetch_ucirepo

    print("   ✅ Library imported successfully.")

    print("3. FETCHING DATASET FROM WEB (Iris, ID=53)...")
    # This tests internet connectivity from within Docker
    start_time = time.time()
    iris = fetch_ucirepo(id=53)
    end_time = time.time()

    print(f"   ✅ Dataset fetched in {end_time - start_time:.2f} seconds.")

    print("4. VERIFYING DATA...")
    # This tests if pandas (a dependency) is handling the data frame correctly
    df = iris.data.features
    print(f"   Dataset Name: {iris.metadata.name}")
    print(f"   Rows loaded:  {df.shape[0]}")
    print(f"   Columns:      {df.shape[1]}")

    print("------------------------------------------------")
    print("🎉 SUCCESS: Docker, Conda, and UCIMLRepo are fully operational.")
    print("------------------------------------------------")

except ImportError as e:
    print(f"   ❌ IMPORT ERROR: {e}")
    print("   Did you rebuild the Docker image after updating environment.yml?")
except Exception as e:
    print(f"   ❌ RUNTIME ERROR: {e}")
