#!/usr/bin/env python3
"""
Performance testing script for BAAI Reranker Service
Tests throughput, latency, and concurrent request handling
"""

import time
import statistics
import concurrent.futures
import requests
import json
import argparse
from typing import List, Dict, Tuple
import random
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for client import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client.reranker_client import RerankerClient, TextPair


@dataclass
class PerformanceMetrics:
    """Store performance test metrics"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    min_latency: float
    max_latency: float
    mean_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float


class PerformanceTester:
    """Performance testing for reranker service"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = RerankerClient(base_url)
        self.test_data = self._generate_test_data()
    
    def _generate_test_data(self) -> Dict:
        """Generate test queries and passages"""
        queries = [
            "What is machine learning?",
            "How does natural language processing work?",
            "Explain deep learning",
            "What are neural networks?",
            "Define artificial intelligence",
            "What is computer vision?",
            "How do transformers work?",
            "What is reinforcement learning?",
            "Explain gradient descent",
            "What are embeddings?"
        ]
        
        passages = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "Natural language processing allows computers to understand and generate human language.",
            "Deep learning uses multi-layer neural networks to learn complex patterns.",
            "Neural networks are computing systems inspired by biological neural networks.",
            "Artificial intelligence is the simulation of human intelligence by machines.",
            "Computer vision enables machines to interpret and understand visual information.",
            "Transformers are a type of neural network architecture based on attention mechanisms.",
            "Reinforcement learning trains agents through rewards and penalties.",
            "Gradient descent is an optimization algorithm for finding local minima.",
            "Embeddings are dense vector representations of discrete objects."
        ]
        
        return {"queries": queries, "passages": passages}
    
    def test_single_requests(self, num_requests: int = 100) -> PerformanceMetrics:
        """Test performance of single rerank requests"""
        print(f"\n📊 Testing {num_requests} single rerank requests...")
        
        latencies = []
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        for i in range(num_requests):
            query = random.choice(self.test_data["queries"])
            passage = random.choice(self.test_data["passages"])
            
            request_start = time.time()
            try:
                self.client.rerank_single(query, passage)
                successful += 1
                latency = time.time() - request_start
                latencies.append(latency)
            except Exception as e:
                failed += 1
                print(f"Request {i+1} failed: {e}")
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{num_requests} requests...")
        
        total_time = time.time() - start_time
        
        return self._calculate_metrics(
            latencies, successful, failed, total_time, num_requests
        )
    
    def test_batch_requests(self, num_requests: int = 50, batch_size: int = 10) -> PerformanceMetrics:
        """Test performance of batch rerank requests"""
        print(f"\n📊 Testing {num_requests} batch requests (batch size: {batch_size})...")
        
        latencies = []
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        for i in range(num_requests):
            # Generate batch
            pairs = []
            for _ in range(batch_size):
                query = random.choice(self.test_data["queries"])
                passage = random.choice(self.test_data["passages"])
                pairs.append(TextPair(query, passage))
            
            request_start = time.time()
            try:
                self.client.rerank_batch(pairs)
                successful += 1
                latency = time.time() - request_start
                latencies.append(latency)
            except Exception as e:
                failed += 1
                print(f"Batch request {i+1} failed: {e}")
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{num_requests} batch requests...")
        
        total_time = time.time() - start_time
        
        return self._calculate_metrics(
            latencies, successful, failed, total_time, num_requests
        )
    
    def test_concurrent_requests(self, num_workers: int = 10, requests_per_worker: int = 10) -> PerformanceMetrics:
        """Test performance with concurrent requests"""
        print(f"\n📊 Testing concurrent requests ({num_workers} workers, {requests_per_worker} requests each)...")
        
        def worker_task(worker_id: int) -> List[float]:
            """Task for each worker thread"""
            worker_latencies = []
            for _ in range(requests_per_worker):
                query = random.choice(self.test_data["queries"])
                passage = random.choice(self.test_data["passages"])
                
                request_start = time.time()
                try:
                    with RerankerClient(self.base_url) as client:
                        client.rerank_single(query, passage)
                    latency = time.time() - request_start
                    worker_latencies.append(latency)
                except Exception as e:
                    print(f"Worker {worker_id} request failed: {e}")
            
            return worker_latencies
        
        start_time = time.time()
        all_latencies = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_workers)]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    worker_latencies = future.result()
                    all_latencies.extend(worker_latencies)
                except Exception as e:
                    print(f"Worker failed: {e}")
        
        total_time = time.time() - start_time
        total_requests = num_workers * requests_per_worker
        successful = len(all_latencies)
        failed = total_requests - successful
        
        return self._calculate_metrics(
            all_latencies, successful, failed, total_time, total_requests
        )
    
    def test_stress(self, duration_seconds: int = 60, target_rps: int = 10):
        """Stress test with sustained load"""
        print(f"\n📊 Stress testing for {duration_seconds}s at {target_rps} RPS...")
        
        interval = 1.0 / target_rps
        latencies = []
        successful = 0
        failed = 0
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        while time.time() < end_time:
            query = random.choice(self.test_data["queries"])
            passage = random.choice(self.test_data["passages"])
            
            request_start = time.time()
            try:
                self.client.rerank_single(query, passage)
                successful += 1
                latency = time.time() - request_start
                latencies.append(latency)
            except Exception as e:
                failed += 1
            
            # Rate limiting
            elapsed = time.time() - request_start
            if elapsed < interval:
                time.sleep(interval - elapsed)
            
            if len(latencies) % 50 == 0:
                elapsed_total = time.time() - start_time
                print(f"  {elapsed_total:.1f}s: {len(latencies)} requests completed...")
        
        total_time = time.time() - start_time
        total_requests = successful + failed
        
        return self._calculate_metrics(
            latencies, successful, failed, total_time, total_requests
        )
    
    def _calculate_metrics(
        self,
        latencies: List[float],
        successful: int,
        failed: int,
        total_time: float,
        total_requests: int
    ) -> PerformanceMetrics:
        """Calculate performance metrics from test results"""
        if not latencies:
            return PerformanceMetrics(
                total_requests=total_requests,
                successful_requests=successful,
                failed_requests=failed,
                total_time=total_time,
                min_latency=0,
                max_latency=0,
                mean_latency=0,
                median_latency=0,
                p95_latency=0,
                p99_latency=0,
                throughput=0
            )
        
        sorted_latencies = sorted(latencies)
        
        return PerformanceMetrics(
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            total_time=total_time,
            min_latency=min(latencies),
            max_latency=max(latencies),
            mean_latency=statistics.mean(latencies),
            median_latency=statistics.median(latencies),
            p95_latency=sorted_latencies[int(len(sorted_latencies) * 0.95)] if len(sorted_latencies) > 0 else 0,
            p99_latency=sorted_latencies[int(len(sorted_latencies) * 0.99)] if len(sorted_latencies) > 0 else 0,
            throughput=successful / total_time if total_time > 0 else 0
        )
    
    def print_metrics(self, metrics: PerformanceMetrics, test_name: str):
        """Print formatted performance metrics"""
        print(f"\n✅ {test_name} Results:")
        print("=" * 50)
        print(f"Total Requests:     {metrics.total_requests}")
        print(f"Successful:         {metrics.successful_requests}")
        print(f"Failed:             {metrics.failed_requests}")
        print(f"Success Rate:       {(metrics.successful_requests/metrics.total_requests*100):.1f}%")
        print(f"Total Time:         {metrics.total_time:.2f}s")
        print(f"Throughput:         {metrics.throughput:.2f} req/s")
        print("\nLatency Statistics (seconds):")
        print(f"  Min:              {metrics.min_latency:.3f}s")
        print(f"  Max:              {metrics.max_latency:.3f}s")
        print(f"  Mean:             {metrics.mean_latency:.3f}s")
        print(f"  Median:           {metrics.median_latency:.3f}s")
        print(f"  95th percentile:  {metrics.p95_latency:.3f}s")
        print(f"  99th percentile:  {metrics.p99_latency:.3f}s")
    
    def run_all_tests(self):
        """Run all performance tests"""
        print("\n🚀 Starting Performance Test Suite")
        print("=" * 60)
        
        # Check service health
        try:
            health = self.client.health_check()
            print(f"✅ Service is healthy: {health['status']}")
        except Exception as e:
            print(f"❌ Service health check failed: {e}")
            return
        
        # Run tests
        tests = [
            ("Single Requests", lambda: self.test_single_requests(100)),
            ("Batch Requests (10 pairs)", lambda: self.test_batch_requests(50, 10)),
            ("Concurrent Requests", lambda: self.test_concurrent_requests(10, 10)),
            ("Stress Test (30s)", lambda: self.test_stress(30, 5))
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                metrics = test_func()
                self.print_metrics(metrics, test_name)
                results[test_name] = metrics
            except Exception as e:
                print(f"\n❌ {test_name} failed: {e}")
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE TEST SUMMARY")
        print("=" * 60)
        
        for test_name, metrics in results.items():
            print(f"\n{test_name}:")
            print(f"  Throughput: {metrics.throughput:.2f} req/s")
            print(f"  Mean Latency: {metrics.mean_latency:.3f}s")
            print(f"  P95 Latency: {metrics.p95_latency:.3f}s")


def main():
    parser = argparse.ArgumentParser(description="Performance testing for BAAI Reranker Service")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the reranker service"
    )
    parser.add_argument(
        "--test",
        choices=["all", "single", "batch", "concurrent", "stress"],
        default="all",
        help="Type of test to run"
    )
    
    args = parser.parse_args()
    
    tester = PerformanceTester(args.url)
    
    if args.test == "all":
        tester.run_all_tests()
    elif args.test == "single":
        metrics = tester.test_single_requests(100)
        tester.print_metrics(metrics, "Single Requests Test")
    elif args.test == "batch":
        metrics = tester.test_batch_requests(50, 10)
        tester.print_metrics(metrics, "Batch Requests Test")
    elif args.test == "concurrent":
        metrics = tester.test_concurrent_requests(10, 10)
        tester.print_metrics(metrics, "Concurrent Requests Test")
    elif args.test == "stress":
        metrics = tester.test_stress(60, 10)
        tester.print_metrics(metrics, "Stress Test")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")