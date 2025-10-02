import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock FlagEmbedding before importing app
with patch('app.main.FlagReranker') as MockReranker:
    mock_instance = Mock()
    mock_instance.compute_score.return_value = -5.0
    MockReranker.return_value = mock_instance
    from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check_success(self):
        """Test successful health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] == True
        assert "model_name" in data
        assert "version" in data


class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns service info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "model" in data
        assert "endpoints" in data


class TestSingleRerankEndpoint:
    """Test single rerank endpoint"""
    
    def test_single_rerank_success(self):
        """Test successful single reranking"""
        with patch('app.main.reranker_model') as mock_model:
            mock_model.compute_score.return_value = -5.0
            
            payload = {
                "query": "What is Python?",
                "passage": "Python is a programming language",
                "normalize": False
            }
            
            response = client.post("/rerank", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "score" in data
            assert data["normalized"] == False
            assert data["query_length"] == len(payload["query"])
            assert data["passage_length"] == len(payload["passage"])
    
    def test_single_rerank_normalized(self):
        """Test single reranking with normalization"""
        with patch('app.main.reranker_model') as mock_model:
            mock_model.compute_score.return_value = 0.9948
            
            payload = {
                "query": "What is Python?",
                "passage": "Python is a programming language",
                "normalize": True
            }
            
            response = client.post("/rerank", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["normalized"] == True
            assert 0 <= data["score"] <= 1
    
    def test_single_rerank_empty_text(self):
        """Test reranking with empty text"""
        payload = {
            "query": "",
            "passage": "Some passage",
            "normalize": False
        }
        
        response = client.post("/rerank", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_single_rerank_missing_field(self):
        """Test reranking with missing field"""
        payload = {
            "query": "What is Python?"
            # Missing passage field
        }
        
        response = client.post("/rerank", json=payload)
        assert response.status_code == 422


class TestBatchRerankEndpoint:
    """Test batch rerank endpoint"""
    
    def test_batch_rerank_success(self):
        """Test successful batch reranking"""
        with patch('app.main.reranker_model') as mock_model:
            mock_model.compute_score.return_value = [-5.0, 3.0]
            
            payload = {
                "pairs": [
                    {"query": "Query 1", "passage": "Passage 1"},
                    {"query": "Query 2", "passage": "Passage 2"}
                ],
                "normalize": False
            }
            
            response = client.post("/rerank/batch", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "scores" in data
            assert len(data["scores"]) == 2
            assert data["normalized"] == False
            assert data["pairs_count"] == 2
    
    def test_batch_rerank_single_pair(self):
        """Test batch reranking with single pair"""
        with patch('app.main.reranker_model') as mock_model:
            mock_model.compute_score.return_value = [-5.0]
            
            payload = {
                "pairs": [
                    {"query": "Query 1", "passage": "Passage 1"}
                ],
                "normalize": False
            }
            
            response = client.post("/rerank/batch", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert len(data["scores"]) == 1
            assert data["pairs_count"] == 1
    
    def test_batch_rerank_normalized(self):
        """Test batch reranking with normalization"""
        with patch('app.main.reranker_model') as mock_model:
            mock_model.compute_score.return_value = [0.9948, 0.0027]
            
            payload = {
                "pairs": [
                    {"query": "Q1", "passage": "P1"},
                    {"query": "Q2", "passage": "P2"}
                ],
                "normalize": True
            }
            
            response = client.post("/rerank/batch", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["normalized"] == True
            for score in data["scores"]:
                assert 0 <= score <= 1
    
    def test_batch_rerank_empty_pairs(self):
        """Test batch reranking with empty pairs list"""
        payload = {
            "pairs": [],
            "normalize": False
        }
        
        response = client.post("/rerank/batch", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_batch_rerank_invalid_pair(self):
        """Test batch reranking with invalid pair"""
        payload = {
            "pairs": [
                {"query": "", "passage": "Valid passage"}  # Empty query
            ],
            "normalize": False
        }
        
        response = client.post("/rerank/batch", json=payload)
        assert response.status_code == 422


class TestErrorHandling:
    """Test error handling"""
    
    def test_model_not_loaded(self):
        """Test response when model is not loaded"""
        with patch('app.main.reranker_model', None):
            response = client.get("/health")
            assert response.status_code == 503
            
            payload = {
                "query": "Test",
                "passage": "Test",
                "normalize": False
            }
            response = client.post("/rerank", json=payload)
            assert response.status_code == 503
    
    def test_invalid_content_type(self):
        """Test invalid content type"""
        response = client.post(
            "/rerank",
            data="not json",
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 422
    
    def test_too_long_text(self):
        """Test text exceeding maximum length"""
        payload = {
            "query": "a" * 10001,  # Exceeds max length
            "passage": "Valid passage",
            "normalize": False
        }
        
        response = client.post("/rerank", json=payload)
        assert response.status_code == 422


class TestCORS:
    """Test CORS configuration"""
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.options(
            "/rerank",
            headers={"Origin": "http://example.com"}
        )
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])