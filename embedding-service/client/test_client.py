#!/usr/bin/env python3
"""
Test client for BAAI BGE-M3 Embedding Service
"""

import requests
import json
import time
import sys

def test_embedding_service(base_url="http://localhost:8001"):
    """Test the BGE-M3 embedding service"""

    print("Testing BAAI BGE-M3 Embedding Service")
    print("=" * 50)

    # Test 1: Health check
    print("\n1. Health Check:")
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        if response.status_code == 200:
            health = response.json()
            print(f"[OK] Service Status: {health['status']}")
            print(f"   Model Loaded: {health['model_loaded']}")
            print(f"   Device: {health['device']}")
            print(f"   Model: {health['model_name']}")
            print(f"   Dimensions: {health['dimensions']}")
        else:
            print(f"[FAIL] Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Health check error: {e}")
        return False

    # Test 2: Single text embedding
    print("\n2. Single Text Embedding:")
    try:
        payload = {
            "texts": ["Machine learning is a subset of artificial intelligence."]
        }

        response = requests.post(f"{base_url}/embed", json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            print(f"[OK] Embedding generated")
            print(f"   Dimensions: {result['dimensions']}")
            print(f"   Texts count: {result['texts_count']}")
            print(f"   First 5 values: {result['embeddings'][0][:5]}")
        else:
            print(f"[FAIL] Single embedding failed: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Single embedding error: {e}")

    # Test 3: Batch embeddings
    print("\n3. Batch Embeddings (5 texts):")
    try:
        payload = {
            "texts": [
                "Python is a high-level programming language.",
                "Machine learning models require large datasets.",
                "Natural language processing is a subfield of AI.",
                "Deep learning uses neural networks.",
                "Data science combines statistics and programming."
            ]
        }

        response = requests.post(f"{base_url}/embed", json=payload, timeout=90)
        if response.status_code == 200:
            result = response.json()
            print(f"[OK] Batch embeddings generated")
            print(f"   Dimensions: {result['dimensions']}")
            print(f"   Texts count: {result['texts_count']}")
            for i in range(len(result['embeddings'])):
                print(f"   Text {i+1} embedding sample: {result['embeddings'][i][:3]}")
        else:
            print(f"[FAIL] Batch embedding failed: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] Batch embedding error: {e}")

    # Test 4: German tax law embeddings (20 passages)
    print("\n4. German Tax Law Embeddings (20 passages):")
    try:
        payload = {
            "texts": [
                "Die Abschreibung von geringwertigen Wirtschaftsgütern gemäß § 6 Abs. 2 EStG ermöglicht Selbständigen den sofortigen Betriebsausgabenabzug.",
                "Der Investitionsabzugsbetrag nach § 7g EStG ist ein erheblicher Steuervorteil für Selbständige.",
                "Die Sonderabschreibung gemäß § 7g Abs. 5 EStG erlaubt Selbständigen eine zusätzliche Abschreibung von bis zu 20%.",
                "Selbständige können bei der Abschreibung von Betriebsmitteln zwischen linearer und degressiver Abschreibung wählen.",
                "Die AfA-Tabellen des Bundesfinanzministeriums legen für verschiedene Betriebsmittel unterschiedliche Nutzungsdauern fest.",
                "Poolabschreibung: Selbständige können seit 2018 alle Wirtschaftsgüter mit Anschaffungskosten zwischen 250 und 1.000 Euro in einen Sammelposten einstellen.",
                "Sofortabschreibung für digitale Wirtschaftsgüter: Das Jahressteuergesetz 2020 hat die Abschreibungsdauer für Computer auf ein Jahr verkürzt.",
                "Bei gemischt genutzten Betriebsmitteln können Selbständige den betrieblichen Nutzungsanteil abschreiben.",
                "Selbständige mit einem Gewinn unter 200.000 Euro können den Investitionsabzugsbetrag ohne Nachweis in Anspruch nehmen.",
                "Für bewegliche Wirtschaftsgüter des Anlagevermögens können Selbständige auch außerplanmäßige Abschreibungen vornehmen.",
                "Die Teilwertabschreibung ermöglicht es Selbständigen, den gesunkenen Wert von Betriebsmitteln steuerlich zu berücksichtigen.",
                "Selbständige können bei Anschaffung eines PKW als Betriebsmittel zwischen verschiedenen Abschreibungsvarianten wählen.",
                "Leasing von Betriebsmitteln bietet Selbständigen einen alternativen Steuervorteil.",
                "Bei Anschaffung gebrauchter Betriebsmittel können Selbständige die Restnutzungsdauer individuell schätzen.",
                "Selbständige in der Existenzgründungsphase profitieren besonders vom Investitionsabzugsbetrag.",
                "Die Umsatzsteuervoranmeldung spielt eine wichtige Rolle bei Betriebsmittelanschaffungen.",
                "Renovierungs- und Modernisierungskosten bei bestehenden Betriebsmitteln können oft als sofort abziehbare Erhaltungsaufwendungen behandelt werden.",
                "Bei Veräußerung oder Entnahme von Betriebsmitteln müssen Selbständige den Restbuchwert gewinnerhöhend auflösen.",
                "Freiberufler und Gewerbetreibende haben unterschiedliche Pflichten bei der Dokumentation von Abschreibungen.",
                "Die Dokumentation der Anschaffung und Nutzung von Betriebsmitteln ist entscheidend für die Anerkennung der Abschreibung."
            ]
        }

        start_time = time.time()
        response = requests.post(f"{base_url}/embed", json=payload, timeout=120)
        elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            print(f"[OK] German tax law embeddings generated")
            print(f"   Dimensions: {result['dimensions']}")
            print(f"   Texts count: {result['texts_count']}")
            print(f"   Processing time: {elapsed:.2f}s")
            print(f"   Average time per text: {elapsed/20:.2f}s")
        else:
            print(f"[FAIL] German embeddings failed: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] German embeddings error: {e}")

    # Test 5: Performance test
    print("\n5. Performance Test:")
    try:
        payload = {
            "texts": ["This is a test sentence for performance measurement."]
        }

        start_time = time.time()
        num_requests = 5

        for i in range(num_requests):
            response = requests.post(f"{base_url}/embed", json=payload, timeout=30)
            if response.status_code != 200:
                print(f"[FAIL] Request {i+1} failed")
                break
        else:
            elapsed = time.time() - start_time
            avg_time = elapsed / num_requests
            print(f"[OK] {num_requests} requests completed in {elapsed:.2f}s")
            print(f"   Average time per request: {avg_time:.3f}s")
    except Exception as e:
        print(f"[FAIL] Performance test error: {e}")

    print("\n" + "=" * 50)
    print("SUCCESS: Testing completed!")

    return True

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test BAAI BGE-M3 embedding service")
    parser.add_argument(
        "--url",
        default="http://localhost:8001",
        help="Base URL of the service (default: http://localhost:8001)"
    )

    args = parser.parse_args()

    # Wait for service to be ready
    print("Waiting for service to be ready...")
    for attempt in range(12):  # Wait up to 2 minutes
        try:
            response = requests.get(f"{args.url}/health", timeout=10)
            if response.status_code == 200:
                print("Service is ready!")
                break
        except:
            pass

        print(f"Attempt {attempt + 1}/12 - waiting...")
        time.sleep(10)
    else:
        print("[FAIL] Service failed to become ready")
        sys.exit(1)

    # Run tests
    success = test_embedding_service(args.url)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
