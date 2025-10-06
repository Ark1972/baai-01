#!/usr/bin/env python3
"""
Test client for Ollama-based BAAI Reranker Service
"""

import requests
import json
import time
import sys
import os

def test_ollama_reranker(base_url="http://localhost:8000"):
    """Test the Ollama-based reranker service"""
    
    print("🔍 Testing Ollama-based BAAI Reranker Service")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Health Check:")
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Service Status: {health['status']}")
            print(f"   Ollama Status: {health['ollama_status']}")
            print(f"   Model: {health['model_name']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Single reranking
    print("\n2. Single Text Pair Reranking:")
    try:
        payload = {
            "query": "What is machine learning?",
            "passage": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "normalize": False
        }
        
        response = requests.post(f"{base_url}/rerank", json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Raw Score: {result['score']:.4f}")
            print(f"   Query Length: {result['query_length']}")
            print(f"   Passage Length: {result['passage_length']}")
        else:
            print(f"❌ Single rerank failed: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Single rerank error: {e}")
    
    # Test 3: Single reranking with normalization
    print("\n3. Single Reranking (Normalized):")
    try:
        payload = {
            "query": "What is machine learning?",
            "passage": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "normalize": True
        }
        
        response = requests.post(f"{base_url}/rerank", json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Normalized Score: {result['score']:.4f} (0-1 range)")
        else:
            print(f"❌ Normalized rerank failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Normalized rerank error: {e}")
    
    # Test 4: Batch reranking
    print("\n4. Batch Reranking:")
    try:
        payload = {
            "pairs": [
                {
                    "query": "What is Python?",
                    "passage": "Python is a high-level programming language known for its simplicity and readability."
                },
                {
                    "query": "What is Python?", 
                    "passage": "A python is a large non-venomous snake found in Africa, Asia, and Australia."
                },
                {
                    "query": "How to cook pasta?",
                    "passage": "To cook pasta, boil water, add salt, add pasta, and cook for 8-12 minutes until al dente."
                },
                {
                    "query": "How to cook pasta?",
                    "passage": "Python is a programming language used for web development and data science."
                }
            ],
            "normalize": True
        }
        
        response = requests.post(f"{base_url}/rerank/batch", json=payload, timeout=90)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch processing successful:")
            print(f"   Pairs processed: {result['pairs_count']}")
            print(f"   Normalized: {result['normalized']}")
            for i, score in enumerate(result['scores']):
                query = payload['pairs'][i]['query'][:30] + "..."
                passage = payload['pairs'][i]['passage'][:40] + "..."
                print(f"   {i+1}. {query} -> {passage} : {score:.4f}")
        else:
            print(f"❌ Batch rerank failed: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Batch rerank error: {e}")
    
    # Test 5: Performance test
    print("\n5. Performance Test:")
    try:
        payload = {
            "query": "artificial intelligence",
            "passage": "AI is transforming technology",
            "normalize": False
        }
        
        start_time = time.time()
        num_requests = 5
        
        for i in range(num_requests):
            response = requests.post(f"{base_url}/rerank", json=payload, timeout=30)
            if response.status_code != 200:
                print(f"❌ Request {i+1} failed")
                break
        else:
            elapsed = time.time() - start_time
            avg_time = elapsed / num_requests
            print(f"✅ {num_requests} requests completed in {elapsed:.2f}s")
            print(f"   Average time per request: {avg_time:.3f}s")
    except Exception as e:
        print(f"❌ Performance test error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Testing completed!")
    
    return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Ollama-based reranker service")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the service (default: http://localhost:8000)"
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
        print("❌ Service failed to become ready")
        sys.exit(1)
    
    # Run tests
    success = test_ollama_reranker(args.url)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()