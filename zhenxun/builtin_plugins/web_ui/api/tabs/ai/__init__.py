from fastapi import APIRouter

from .characters import router as characters_router
from .chat import router as chat_router
from .configuration import router as configuration_router
from .console import router as console_router
from .dsh import router as dsh_router
from .knowledge import router as knowledge_router
from .mcp import router as mcp_router
from .memory import router as memory_router
from .proactive import router as proactive_router
from .probe import router as probe_router
from .sandbox import router as sandbox_router
from .sessions import router as sessions_router
from .social import router as social_router
from .stats import router as stats_router

router = APIRouter(prefix="/ai")
router.include_router(configuration_router)
router.include_router(probe_router)
router.include_router(chat_router)
router.include_router(proactive_router)
router.include_router(sessions_router)
router.include_router(sandbox_router)
router.include_router(console_router)
router.include_router(dsh_router)
router.include_router(stats_router)
router.include_router(social_router)
router.include_router(knowledge_router)
router.include_router(memory_router)
router.include_router(characters_router)
router.include_router(mcp_router)

__all__ = ["router"]
