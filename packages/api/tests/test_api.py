import pytest
from httpx import AsyncClient, ASGITransport
from sqlmodel import SQLModel
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from omniscient_api.app import create_app
from omniscient_api.database import get_session
from omniscient_api.models import AnalysisRequest
from unittest.mock import MagicMock, patch

# Setup in-memory DB for testing
DATABASE_URL = "sqlite+aiosqlite:///:memory:"
engine = create_async_engine(DATABASE_URL, echo=True, poolclass=StaticPool)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

async def get_session_override():
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session

@pytest.fixture
async def client():
    await init_db()
    
    # Mock RAG components to avoid external dependencies
    with patch("omniscient_api.app.OllamaProvider"), \
         patch("omniscient_api.app.RAGPipeline") as MockPipeline:
        
        mock_pipeline = MagicMock()
        MockPipeline.create.return_value = mock_pipeline
        mock_pipeline.initialize = pytest.AsyncMock()
        mock_pipeline.close = pytest.AsyncMock()
        
        app = create_app()
        app.dependency_overrides[get_session] = get_session_override
        
        # Manually set pipeline in state since lifespan might be tricky with mocks in tests sometimes
        app.state.rag_pipeline = mock_pipeline
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

@pytest.mark.asyncio
async def test_create_analysis(client):
    response = await client.post(
        "/api/v1/analyze",
        json={
            "repository_url": "https://github.com/test/repo",
            "agents": ["architecture"]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["repository_url"] == "https://github.com/test/repo"
    assert "analysis_id" in data
    assert data["status"] == "pending"
    
    analysis_id = data["analysis_id"]
    
    # Check status
    response = await client.get(f"/api/v1/analysis/{analysis_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["analysis_id"] == analysis_id

@pytest.mark.asyncio
async def test_list_analyses(client):
    response = await client.get("/api/v1/analyses")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

@pytest.mark.asyncio
async def test_rag_search(client):
    # Mock the pipeline query result
    app = client._transport.app
    app.state.rag_pipeline.query = pytest.AsyncMock(return_value=[])
    
    response = await client.post(
        "/api/v1/rag/search",
        json={
            "query": "test query",
            "top_k": 3
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
