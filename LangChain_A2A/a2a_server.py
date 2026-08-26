import uvicorn
from starlette.applications import Starlette

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.helpers import new_text_message
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils import AGENT_CARD_WELL_KNOWN_PATH, DEFAULT_RPC_URL, TransportProtocol
from a2a.utils.errors import UnsupportedOperationError


# ============================================================
# 1. 에이전트 로직 계층
# ============================================================
class HelloWorldAgent:
    """고정된 인사말을 반환하는 에이전트"""

    async def invoke(self) -> str:
        # 입력과 무관하게 항상 동일한 문자열을 반환
        return "Hello World! 안녕하세요, A2A 에이전트입니다!"


# ============================================================
# 2. A2A 프로토콜 어댑터 계층 (AgentExecutor)
# ============================================================
class HelloWorldAgentExecutor(AgentExecutor):
    """A2A 요청을 받아 HelloWorldAgent를 호출하는 실행자"""

    def __init__(self):
        # 에이전트 인스턴스를 생성하여 보관
        self.agent = HelloWorldAgent()

    async def execute(
        self,
        context: RequestContext,   # 요청 컨텍스트 (사용자 메시지, task_id, context_id 보관)
        event_queue: EventQueue,   # 이 큐에 넣은 이벤트가 클라이언트 응답이 됨
    ) -> None:
        # 에이전트 로직을 실행하여 문자열 결과를 획득
        result = await self.agent.invoke()

        # 문자열을 role=ROLE_AGENT인 A2A Message로 변환한 뒤 큐에 적재
        await event_queue.enqueue_event(new_text_message(result))

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # 취소 미지원을 A2A 표준 에러로 반환
        raise UnsupportedOperationError()


# ============================================================
# 3. 에이전트 카드 정의
# ============================================================
def create_agent_card() -> AgentCard:
    """에이전트의 메타데이터를 담은 AgentCard를 생성"""

    # AgentSkill: 에이전트가 수행 가능한 개별 작업 단위
    skill = AgentSkill(
        id='hello_world',                      # 스킬 고유 식별자
        name='Hello World 인사',                # 스킬 표시 이름
        description='간단한 인사말을 반환합니다',  # 스킬 설명
        tags=['인사', 'hello world', '기본'],    # 검색/라우팅용 키워드
        examples=['안녕', '안녕하세요', 'hi', 'hello'],  # 호출 예시 프롬프트
    )

    # AgentCapabilities: 프로토콜 수준의 기능 지원 여부 플래그
    capabilities = AgentCapabilities(
        streaming=False,           # SSE 스트리밍 응답 지원 여부
        push_notifications=False,  # 웹훅 푸시 알림 지원 여부
    )

    # AgentInterface: 접속 주소와 전송 바인딩을 정의 (1.0에서 신설)
    interface = AgentInterface(
        url='http://localhost:9999/',                     # JSON-RPC 요청을 받을 주소
        protocol_binding=TransportProtocol.JSONRPC.value, # 전송 바인딩 종류
        protocol_version='1.0',                           # A2A 프로토콜 버전
    )

    # AgentCard: 클라이언트가 조회하는 에이전트 명세서
    agent_card = AgentCard(
        name='Hello World 에이전트',
        description='A2A 프로토콜을 학습하기 위한 가장 간단한 에이전트입니다',
        version='1.0.0',
        supported_interfaces=[interface],  # 지원 인터페이스 목록 (url 필드를 대체)
        default_input_modes=['text'],      # 입력 형식
        default_output_modes=['text'],     # 출력 형식
        capabilities=capabilities,
        skills=[skill],
    )

    return agent_card


# ============================================================
# 4. 서버 조립 및 기동
# ============================================================
def main():
    """A2A 에이전트 서버를 구성하고 실행"""

    # 에이전트 카드 생성
    agent_card = create_agent_card()

    # JSON-RPC 메서드를 해석해 실행자를 호출하는 요청 처리기 생성
    request_handler = DefaultRequestHandler(
        agent_executor=HelloWorldAgentExecutor(),
        task_store=InMemoryTaskStore(),  # 태스크를 메모리에 보관하는 저장소
        agent_card=agent_card,           # 1.0에서 필수 인자로 추가됨
    )

    # GET /.well-known/agent-card.json 라우트 생성
    routes = create_agent_card_routes(
        agent_card=agent_card,
        card_url=AGENT_CARD_WELL_KNOWN_PATH,
    )

    # POST / JSON-RPC 라우트 생성 후 기존 라우트 목록에 추가
    routes += create_jsonrpc_routes(
        request_handler=request_handler,
        rpc_url=DEFAULT_RPC_URL,
    )

    # 생성된 라우트로 Starlette ASGI 앱을 직접 구성
    app = Starlette(routes=routes)

    # uvicorn으로 서빙
    uvicorn.run(app, host='0.0.0.0', port=9999)


if __name__ == '__main__':
    main()