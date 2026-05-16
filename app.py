"""
AI客服对话质量评估器
AI Customer Service Conversation Quality Evaluator

独立作品项目 — 面试演示用
技术栈: Python + Streamlit + plotly + (可选) OpenAI/Claude API
"""

import streamlit as st
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# ═══════════════════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI客服对话质量评估器",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# 评估标准定义（基于携程AI质检实践经验）
# ═══════════════════════════════════════════════════════════

EVALUATION_DIMENSIONS = {
    "识别需求": {
        "weight": 0.25,
        "description": "客服是否准确理解消费者的问题和真实诉求",
        "rubric": {
            5: "不仅识别表面问题，还挖掘了深层需求，消费者感到'被理解'",
            4: "准确识别了主要问题，基本覆盖消费者诉求",
            3: "识别了主要问题但遗漏了部分细节",
            2: "部分理解错误，需要消费者重复说明",
            1: "完全理解错误或未尝试理解消费者问题",
        },
    },
    "有效共情": {
        "weight": 0.25,
        "description": "客服是否展现了恰当的同理心",
        "rubric": {
            5: "共情自然恰当、时机准确，有效缓解消费者负面情绪",
            4: "有共情表达，基本恰当但略显模板化",
            3: "有共情尝试但时机或方式不够好",
            2: "共情缺乏或过度夸张显得虚假",
            1: "完全冷漠、机械化，或态度恶劣",
        },
    },
    "达成一致": {
        "weight": 0.30,
        "description": "客服是否给出了消费者可接受的解决方案",
        "rubric": {
            5: "方案明确可行且消费者明确表示接受，有备选方案",
            4: "给出了可行方案，消费者基本接受",
            3: "有方案但不够明确，消费者态度模糊",
            2: "方案不合理或消费者明显不接受",
            1: "未提供任何有效解决方案或直接拒绝",
        },
    },
    "承诺回复": {
        "weight": 0.20,
        "description": "客服是否明确了后续步骤和时间节点",
        "rubric": {
            5: "清晰告知下一步动作、明确时间节点、对接人，消费者知道接下来会发生什么",
            4: "告知了下一步但不明确时间节点",
            3: "承诺了会处理但缺乏具体信息",
            2: "含糊其辞，消费者不确定是否会被跟进",
            1: "没有任何承诺或直接结束对话",
        },
    },
}

# ═══════════════════════════════════════════════════════════
# 规则引擎评估（无需API，基于关键词和模式匹配）
# ═══════════════════════════════════════════════════════════

NEED_RECOGNITION_POSITIVE = [
    "您是说", "我理解您", "您的问题是", "您的意思是", "帮您确认", "核实一下",
    "让我看看", "帮您查", "明白了", "了解了", "知道您的情况",
]
NEED_RECOGNITION_NEGATIVE = [
    "没听懂", "再说一遍", "什么意思", "不知道您在说什么", "我没理解",
]

EMPATHY_POSITIVE = [
    "理解您的心情", "非常抱歉", "给您带来不便", "确实是我们不好", "让您久等了",
    "我明白您的感受", "请您放心", "我会尽力", "感谢您的耐心", "辛苦您了",
]
EMPATHY_NEGATIVE = [
    "这是规定", "我也没办法", "不是我负责", "你去找", "不关我事", "你不满意我也没办法",
]

RESOLUTION_POSITIVE = [
    "帮您申请", "为您办理", "给您退款", "帮您协调", "已为您提交",
    "补偿方案", "优惠券", "退差价", "换货处理", "加急处理",
    "您看这样行吗", "您觉得可以吗", "这个方案您接受吗",
]
RESOLUTION_NEGATIVE = [
    "做不到", "不行", "不可能", "没有办法", "不能退", "我们没错",
    "你自己弄的", "跟我们没关系",
]

COMMITMENT_POSITIVE = [
    "X个工作日内", "明天", "今天", "稍后", "马上", "会尽快", "预计",
    "我会跟进", "会有专人", "请注意查收", "会有通知", "会联系您",
    "我这边帮您", "后续您可以看到",
]
COMMITMENT_NEGATIVE = [
    "等着吧", "不一定", "不敢保证", "说不准", "看情况", "到时候再说",
]


def rule_based_evaluate(conversation):
    """基于规则引擎的对话质量评估（无需API）"""
    if not isinstance(conversation, str) or not conversation.strip():
        return {dim: {"score": 3, "reason": "对话内容不足，默认评分"} for dim in EVALUATION_DIMENSIONS}

    text = conversation.lower()

    scores = {}
    reasons = {}

    # 维度1：识别需求
    pos_hits_1 = sum(1 for kw in NEED_RECOGNITION_POSITIVE if kw in text)
    neg_hits_1 = sum(1 for kw in NEED_RECOGNITION_NEGATIVE if kw in text)
    score_1 = min(5, max(1, 3 + pos_hits_1 * 0.4 - neg_hits_1 * 0.8))
    scores["识别需求"] = round(score_1)
    reasons["识别需求"] = f"识别出{pos_hits_1}个正向信号、{neg_hits_1}个负向信号"

    # 维度2：有效共情
    pos_hits_2 = sum(1 for kw in EMPATHY_POSITIVE if kw in text)
    neg_hits_2 = sum(1 for kw in EMPATHY_NEGATIVE if kw in text)
    score_2 = min(5, max(1, 3 + pos_hits_2 * 0.3 - neg_hits_2 * 1.0))
    scores["有效共情"] = round(score_2)
    reasons["有效共情"] = f"识别出{pos_hits_2}个正向信号、{neg_hits_2}个负向信号"

    # 维度3：达成一致
    pos_hits_3 = sum(1 for kw in RESOLUTION_POSITIVE if kw in text)
    neg_hits_3 = sum(1 for kw in RESOLUTION_NEGATIVE if kw in text)
    score_3 = min(5, max(1, 3 + pos_hits_3 * 0.3 - neg_hits_3 * 1.0))
    scores["达成一致"] = round(score_3)
    reasons["达成一致"] = f"识别出{pos_hits_3}个正向信号、{neg_hits_3}个负向信号"

    # 维度4：承诺回复
    pos_hits_4 = sum(1 for kw in COMMITMENT_POSITIVE if kw in text)
    neg_hits_4 = sum(1 for kw in COMMITMENT_NEGATIVE if kw in text)
    score_4 = min(5, max(1, 3 + pos_hits_4 * 0.3 - neg_hits_4 * 1.0))
    scores["承诺回复"] = round(score_4)
    reasons["承诺回复"] = f"识别出{pos_hits_4}个正向信号、{neg_hits_4}个负向信号"

    return {dim: {"score": scores[dim], "reason": reasons[dim]} for dim in EVALUATION_DIMENSIONS}


def llm_evaluate(conversation, api_key=None, model="gpt-3.5-turbo"):
    """基于LLM的对话质量评估（可选增强）"""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        system_prompt = """你是资深客服质检专家。请评估以下客服对话，按四个维度打分(1-5分)，输出JSON。

评分标准：
1. 识别需求：客服是否准确理解消费者问题？5=深刻理解+挖掘深层需求，1=完全理解错误
2. 有效共情：客服是否展现恰当同理心？5=自然恰当有效缓解情绪，1=冷漠机械化
3. 达成一致：客服是否给出可接受方案？5=方案明确可行消费者接受，1=未提供方案
4. 承诺回复：客服是否明确后续步骤？5=清晰告知时间节点和对接人，1=无任何承诺

返回JSON格式：{"识别需求": {"score": 5, "reason": "..."}, ...}"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请评估以下客服对话：\n\n{conversation}"},
            ],
            temperature=0.1,
        )

        result = json.loads(response.choices[0].message.content)
        return result
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════
# 评估结果处理
# ═══════════════════════════════════════════════════════════

def calculate_total_score(dim_scores):
    """计算加权总分"""
    total = 0
    for dim, config in EVALUATION_DIMENSIONS.items():
        weight = config["weight"]
        score = dim_scores[dim]["score"]
        total += score * weight
    return round(total, 1)


def get_improvement_suggestions(dim_scores):
    """根据低分维度生成改进建议"""
    suggestions = []
    for dim, data in dim_scores.items():
        if data["score"] <= 2:
            suggestions.append({
                "dimension": dim,
                "severity": "严重",
                "suggestion": f"**{dim}**得分{data['score']}分，需重点改进",
                "detail": EVALUATION_DIMENSIONS[dim]["rubric"][data["score"]],
            })
        elif data["score"] == 3:
            suggestions.append({
                "dimension": dim,
                "severity": "一般",
                "suggestion": f"**{dim}**得分{data['score']}分，有提升空间",
                "detail": EVALUATION_DIMENSIONS[dim]["rubric"][3],
            })

    return suggestions


# ═══════════════════════════════════════════════════════════
# 示例数据
# ═══════════════════════════════════════════════════════════

def generate_sample_conversations():
    """生成20段模拟客服对话"""
    conversations = [
        # 高分对话 (5条)
        {
            "conversation_id": "G001",
            "agent_name": "客服A",
            "conversation_text": """消费者：我买的手机收到之后屏幕有一道划痕，我要退货。
客服：非常抱歉给您带来不好的体验！您是说收到的手机屏幕上有一道划痕对吗？我完全理解您的心情，花了钱收到有问题的产品确实让人不舒服。您放心，我马上帮您核实一下。
消费者：对，我刚拆开就看到了，太失望了。
客服：确实是我们工作不到位，您先别着急。我帮您申请换货处理，新的手机会在2个工作日内发出。同时我这边帮您申请一张50元优惠券作为补偿，您看这样可以吗？
消费者：那行吧，快点换就行。
客服：好的，已为您提交换货申请，订单号HC20260517001。稍后您会收到短信确认。预计后天发货，物流单号会同步推送。后续有任何问题可以随时联系我们，我会全程跟进。""",
        },
        {
            "conversation_id": "G002",
            "agent_name": "客服B",
            "conversation_text": """消费者：物流显示已签收但我没收到包裹，怎么回事？
客服：非常抱歉让您遇到这种情况！您的意思是物流显示签收但您实际没有收到包裹对吧？我帮您查一下具体情况。
消费者：对啊，都显示签收两天了，我根本没收到。
客服：我完全理解您的着急，等待了这么久却收不到货确实让人焦虑。我现在就帮您核实物流情况。已联系物流商确认配送细节，同时先帮您申请补发，新的包裹预计明天发出。如果原包裹找到的话您可以直接拒收。这样处理您觉得可以吗？
消费者：那行，尽快吧。
客服：好的，补发订单已生成，单号稍后同步给您。我会在明天上午跟进物流进度并主动联系您同步进展。感谢您的耐心！""",
        },
        {
            "conversation_id": "G003",
            "agent_name": "客服A",
            "conversation_text": """消费者：我买了两件衣服，只收到一件。
客服：非常抱歉给您添麻烦了！您是说买了两件只收到一件对吗？帮您确认一下，订单里是两件商品对吗？
消费者：对，就发了一件，白色的没发。
客服：明白了，白色的那件确实漏发了。非常抱歉是我们仓库的问题。我马上帮您办理补发，预计明天发出，后天就能到。同时帮您申请一张20元优惠券作为补偿，您看可以吗？
消费者：可以，赶紧发吧。
客服：好的，补发已提交。明天下午会有物流更新，请注意查收短信。如果有其他问题随时联系我，我是客服A，工号1001。""",
        },
        {
            "conversation_id": "G004",
            "agent_name": "客服C",
            "conversation_text": """消费者：这个订单我要取消退款。
客服：好的，帮您查看一下。您的订单目前已经发货了，物流已经在中转中。您是确定要取消还是要先了解一下预计到达时间？
消费者：太慢了不想要了。
客服：理解您的心情，等待确实让人着急。我帮您查了一下，您的包裹预计后天就能送达了。如果现在取消，退回去还需要3-5天才能退款。您看要不要再等两天？我可以帮您申请10元延迟补贴。
消费者：那行吧，再等等。
客服：好的，已为您申请延迟补贴，审核通过后会直接到账。包裹预计后天送达，明天物流会有更新，您可以随时查看。如果后天还没收到您再联系我，我帮您优先处理。""",
        },
        {
            "conversation_id": "G005",
            "agent_name": "客服B",
            "conversation_text": """消费者：买的奶粉收到之后发现罐子瘪了，不敢给孩子喝。
客服：非常抱歉！您是说奶粉罐子瘪了对吧？我完全理解，给宝宝的食品容不得任何问题。您请放心，我立即帮您处理。
消费者：对，这种谁敢给孩子喝啊。
客服：确实应该谨慎，您做得对。我马上帮您申请退货退款并重新下单，新的一罐明天即可发出。瘪罐的那罐您不需要退回，我们会安排快递上门取件。同时为您申请一张30元优惠券。您看这样处理可以吗？
消费者：好，尽快处理。
客服：好的，退款预计1-3个工作日到账。重新下单的奶粉明天发出，物流信息会实时更新。我明天会跟进您的订单确保万无一失。""",
        },
        # Badcase (5条)
        {
            "conversation_id": "B001",
            "agent_name": "客服X",
            "conversation_text": """消费者：我买的鞋子穿了两次就开胶了，质量太差了。
客服：这是你自己穿的问题吧，我们卖了这么多都没人反映。
消费者：什么叫我穿的问题？明明就是质量问题。
客服：我们鞋子质量都没问题，是你自己穿法不对。退不了。
消费者：我要投诉！
客服：随便你。""",
        },
        {
            "conversation_id": "B002",
            "agent_name": "客服Y",
            "conversation_text": """消费者：我的快递显示异常，能不能帮我查一下？
客服：你去问快递公司啊，问我干嘛。
消费者：我是在你们平台买的，当然找你们啊。
客服：我们只管卖，物流不归我们管。你自己打电话问。
消费者：那如果丢件了怎么办？
客服：等着吧，快递公司确认了再说。""",
        },
        {
            "conversation_id": "B003",
            "agent_name": "客服X",
            "conversation_text": """消费者：我收到的衣服颜色不对，我要换一件。
客服：图片和实物有色差是正常的。
消费者：这色差也太大了吧，完全不是一个颜色。
客服：那是你显示器的问题。
消费者：你这什么态度？
客服：我就这态度。不满意去找平台。""",
        },
        {
            "conversation_id": "B004",
            "agent_name": "客服Z",
            "conversation_text": """消费者：买的手机用了两天就黑屏了，我要换一个。
客服：有没有摔过？
消费者：没有，正常用的。
客服：那不会啊，我们从来不出这种问题。你是不是进水了？
消费者：没有，就正常用的。
客服：人为损坏我们不管。想换自己掏钱。""",
        },
        {
            "conversation_id": "B005",
            "agent_name": "客服Y",
            "conversation_text": """消费者：我的退款怎么还没到？已经一周了。
客服：系统在处理。
消费者：什么时候能到？
客服：不知道。
消费者：那你能帮我查一下吗？
客服：系统显示的你也看到了，等着就行。
消费者：能不能给个时间？
客服：看情况吧，说不准。""",
        },
        # 中等水平 (10条)
        {
            "conversation_id": "M001",
            "agent_name": "客服D",
            "conversation_text": """消费者：我的订单已经三天了还没发货。
客服：我帮您查一下。您这个订单目前仓库在备货，预计明天可以发出。
消费者：太慢了，能不能快一点。
客服：好的我帮您备注加急了。
消费者：好吧。""",
        },
        {
            "conversation_id": "M002",
            "agent_name": "客服E",
            "conversation_text": """消费者：买的零食有一包破了。
客服：抱歉。您拍个照片给我看一下。
消费者：拍了，你看。
客服：好的我帮您申请退款这个单品。大概3-5个工作日到账。
消费者：好吧。""",
        },
        {
            "conversation_id": "M003",
            "agent_name": "客服D",
            "conversation_text": """消费者：怎么退货？
客服：您在订单详情里点申请退货就行。
消费者：然后呢？
客服：然后选择退货原因，提交，会有快递上门取件。
消费者：运费谁出？
客服：质量问题我们出，非质量问题您出。
消费者：好吧。""",
        },
        {
            "conversation_id": "M004",
            "agent_name": "客服F",
            "conversation_text": """消费者：这个商品能优惠吗？
客服：不好意思这款不参加活动。
消费者：那有没有类似的便宜点的？
客服：您可以看下这个链接，类似的款式。
消费者：好的谢谢。
客服：不客气。""",
        },
        {
            "conversation_id": "M005",
            "agent_name": "客服E",
            "conversation_text": """消费者：物流能不能改地址？
客服：您是已经发货了还是没发货？
消费者：已经发货了。
客服：那需要联系快递公司改地址，我给您电话。
消费者：你不能直接帮我改吗？
客服：已发出的包裹我们系统改不了，需要联系快递。
消费者：好吧发给我。""",
        },
        {
            "conversation_id": "M006",
            "agent_name": "客服F",
            "conversation_text": """消费者：发票怎么申请？
客服：您在订单详情页有个电子发票入口，点进去填写信息就行。
消费者：多久能开好？
客服：一般1-2个工作日。
消费者：好的。""",
        },
        {
            "conversation_id": "M007",
            "agent_name": "客服D",
            "conversation_text": """消费者：收到了但是包装盒坏了。
客服：东西有没有损坏？
消费者：东西倒是好的。
客服：那不影响使用就行，包装我们下次注意。
消费者：嗯。""",
        },
        {
            "conversation_id": "M008",
            "agent_name": "客服E",
            "conversation_text": """消费者：想问下这个有没有保修。
客服：有的，支持全国联保一年。
消费者：怎么保修？
客服：您拿着购买凭证去任意售后网点就行。
消费者：好的知道了。""",
        },
        {
            "conversation_id": "M009",
            "agent_name": "客服F",
            "conversation_text": """消费者：优惠券过期了能补发吗？
客服：过期了系统补不了，不好意思。
消费者：那下次什么时候有活动？
客服：您可以关注店铺首页，不定期有活动。
消费者：好吧。""",
        },
        {
            "conversation_id": "M010",
            "agent_name": "客服D",
            "conversation_text": """消费者：这个码数偏大还是偏小？
客服：正常码，建议按平时尺码选。
消费者：我平时穿L，但这个评论说偏小。
客服：那您选XL比较保险。
消费者：好的谢谢。""",
        },
    ]
    return conversations


# ═══════════════════════════════════════════════════════════
# Streamlit 界面
# ═══════════════════════════════════════════════════════════

def main():
    if "cs_convos" not in st.session_state:
        st.session_state["cs_convos"] = None
    if "cs_results" not in st.session_state:
        st.session_state["cs_results"] = None

    # ---- 顶部 ----
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🎧 AI客服对话质量评估器")
        st.caption("输入客服对话 → 四维度自动评分 → 雷达图可视化 → Badcase清单 → 改进建议")
    with col2:
        st.metric("评估维度", "4维")
        st.metric("评分标尺", "1-5分")

    st.divider()

    # ---- 侧边栏 ----
    with st.sidebar:
        st.header("⚙️ 操作面板")

        mode = st.radio("评估模式", ["📝 单条评估", "📊 批量评估"], key="eval_mode")

        st.divider()

        st.subheader("🤖 评估引擎")
        use_llm = st.checkbox("启用LLM增强评估", value=False, help="使用AI进行更精准的评估，需API Key")
        api_key = None
        if use_llm:
            api_key = st.text_input("OpenAI API Key", type="password")

        st.divider()

        st.subheader("📥 示例数据")
        if st.button("加载20段示例对话", type="primary", use_container_width=True):
            st.session_state["cs_convos"] = generate_sample_conversations()
            st.rerun()

        if st.session_state["cs_convos"] is not None:
            if st.button("🗑️ 清除数据", use_container_width=True):
                st.session_state["cs_convos"] = None
                st.session_state["cs_results"] = None
                st.rerun()

        st.divider()
        st.caption("💡 规则引擎模式下无需API，基于关键词和模式匹配；LLM模式评估更精准")

    # ---- 主区域 ----
    if mode == "📝 单条评估":
        show_single_eval(use_llm, api_key)
    else:
        show_batch_eval(use_llm, api_key)


def show_single_eval(use_llm=False, api_key=None):
    """单条对话评估"""
    st.subheader("📝 单条对话评估")

    conversation = st.text_area(
        "粘贴客服对话文本",
        height=200,
        placeholder="消费者：......\n客服：......\n消费者：......\n客服：......",
        key="single_conv",
    )

    if st.button("🔍 开始评估", type="primary", disabled=not conversation):
        with st.spinner("正在评估..."):
            if use_llm and api_key:
                result = llm_evaluate(conversation, api_key)
                if result is None:
                    st.warning("LLM评估失败，回退到规则引擎")
                    result = rule_based_evaluate(conversation)
            else:
                result = rule_based_evaluate(conversation)

        total = calculate_total_score(result)

        col_r1, col_r2 = st.columns([1, 2])

        with col_r1:
            # 总分展示
            st.markdown(f"### 加权总分")
            color = "#66BB6A" if total >= 4 else "#FFA726" if total >= 3 else "#FF4444"
            st.markdown(f"<h1 style='text-align:center;color:{color}'>{total}/5.0</h1>", unsafe_allow_html=True)

            # 雷达图
            categories = list(EVALUATION_DIMENSIONS.keys())
            values = [result[dim]["score"] for dim in categories]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values, theta=categories, fill="toself",
                name="本次评估", line=dict(color="#2196F3", width=2),
                fillcolor="rgba(33,150,243,0.3)",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[4, 4, 4, 4], theta=categories, fill="none",
                name="优秀线", line=dict(color="#90A4AE", width=1, dash="dash"),
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                height=350,
                margin=dict(l=40, r=40, t=20, b=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_r2:
            st.markdown("### 各维度评分")
            for dim in EVALUATION_DIMENSIONS.keys():
                score = result[dim]["score"]
                reason = result[dim]["reason"]
                color = "#66BB6A" if score >= 4 else "#FFA726" if score >= 3 else "#FF4444"
                st.markdown(f"**{dim}**: :{color}[{score}/5] — {reason}")

            # 改进建议
            suggestions = get_improvement_suggestions(result)
            if suggestions:
                st.markdown("### ⚠️ 改进建议")
                for s in suggestions:
                    if s["severity"] == "严重":
                        st.error(s["suggestion"])
                    else:
                        st.warning(s["suggestion"])


def show_batch_eval(use_llm=False, api_key=None):
    """批量对话评估"""
    st.subheader("📊 批量对话评估")

    # 检查是否有加载的对话数据
    if st.session_state["cs_convos"] is not None:
        conversations = st.session_state["cs_convos"]
    else:
        uploaded_file = st.file_uploader("📤 上传客服对话CSV", type=["csv"],
                                         help="需包含 conversation_text 列")
        if uploaded_file is None:
            st.info("请上传CSV文件或在左侧加载示例数据")
            return

        try:
            df_upload = pd.read_csv(uploaded_file)
            text_col = None
            for col in ["conversation_text", "对话文本", "text", "content"]:
                if col in df_upload.columns:
                    text_col = col
                    break
            if text_col is None:
                st.error("未找到对话文本列")
                return

            conversations = []
            for _, row in df_upload.iterrows():
                conv = {"conversation_id": str(row.get("conversation_id", len(conversations))),
                        "agent_name": str(row.get("agent_name", "未知")),
                        "conversation_text": str(row[text_col])}
                conversations.append(conv)
        except Exception as e:
            st.error(f"读取失败: {e}")
            return

    if not conversations:
        return

    # 执行批量评估
    if st.button("🔍 开始批量评估", type="primary") or st.session_state.get("cs_results") is not None:
        if st.session_state.get("cs_results") is None:
            results = []
            progress_bar = st.progress(0)
            for i, conv in enumerate(conversations):
                if use_llm and api_key:
                    result = llm_evaluate(conv["conversation_text"], api_key)
                    if result is None:
                        result = rule_based_evaluate(conv["conversation_text"])
                else:
                    result = rule_based_evaluate(conv["conversation_text"])

                total = calculate_total_score(result)
                results.append({
                    "conversation_id": conv["conversation_id"],
                    "agent_name": conv["agent_name"],
                    "总分": total,
                    "识别需求": result["识别需求"]["score"],
                    "有效共情": result["有效共情"]["score"],
                    "达成一致": result["达成一致"]["score"],
                    "承诺回复": result["承诺回复"]["score"],
                    "是否Badcase": "是" if total < 3.0 else "否",
                    "conversation_text": conv["conversation_text"],
                })
                progress_bar.progress((i + 1) / len(conversations))
            progress_bar.empty()
            st.session_state["cs_results"] = results
        else:
            results = st.session_state["cs_results"]

        # KPI
        df_results = pd.DataFrame(results)
        badcase_count = len(df_results[df_results["是否Badcase"] == "是"])

        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            st.metric("评估总数", len(results))
        with col_k2:
            avg_total = round(df_results["总分"].mean(), 1)
            st.metric("平均总分", f"{avg_total}/5.0")
        with col_k3:
            st.metric("Badcase数", badcase_count, delta="需关注" if badcase_count > 0 else "正常")
        with col_k4:
            best_agent = df_results.groupby("agent_name")["总分"].mean().idxmax() if "agent_name" in df_results.columns else "-"
            st.metric("最佳客服", best_agent)

        st.divider()

        # Tab页
        tab_b1, tab_b2, tab_b3 = st.tabs(["📋 评估结果", "📊 维度分析", "⚠️ Badcase清单"])

        with tab_b1:
            st.subheader("评估结果明细")
            display_cols = ["conversation_id", "agent_name", "总分", "识别需求", "有效共情", "达成一致", "承诺回复", "是否Badcase"]
            avail_cols = [c for c in display_cols if c in df_results.columns]
            st.dataframe(df_results[avail_cols], use_container_width=True, height=400)

        with tab_b2:
            st.subheader("维度得分分析")

            col_v1, col_v2 = st.columns(2)

            with col_v1:
                # 各维度平均分
                dim_avgs = {dim: round(df_results[dim].mean(), 1) for dim in EVALUATION_DIMENSIONS.keys()}
                fig_dim = px.bar(
                    x=list(dim_avgs.keys()), y=list(dim_avgs.values()),
                    title="各维度平均得分", color=list(dim_avgs.values()),
                    color_continuous_scale="RdYlGn", range_color=[1, 5],
                    labels={"x": "维度", "y": "平均分"},
                )
                fig_dim.update_layout(yaxis_range=[0, 5])
                st.plotly_chart(fig_dim, use_container_width=True)

            with col_v2:
                # 各客服总分配对
                if "agent_name" in df_results.columns:
                    agent_scores = df_results.groupby("agent_name")["总分"].mean().sort_values(ascending=False)
                    fig_agent = px.bar(
                        x=agent_scores.index, y=agent_scores.values,
                        title="客服平均总分配对",
                        color=agent_scores.values,
                        color_continuous_scale="RdYlGn", range_color=[1, 5],
                        labels={"x": "客服", "y": "平均总分"},
                    )
                    fig_agent.update_layout(yaxis_range=[0, 5])
                    st.plotly_chart(fig_agent, use_container_width=True)

            # 分数分布直方图
            fig_hist = px.histogram(
                df_results, x="总分", nbins=10,
                title="总分配布", color_discrete_sequence=["#2196F3"],
            )
            fig_hist.add_vline(x=3.0, line_dash="dash", line_color="red", annotation_text="Badcase线")
            st.plotly_chart(fig_hist, use_container_width=True)

        with tab_b3:
            st.subheader("⚠️ Badcase清单")

            badcase_df = df_results[df_results["是否Badcase"] == "是"].sort_values("总分")

            if badcase_df.empty:
                st.success("✅ 没有Badcase！所有对话总分均>=3.0")
            else:
                st.warning(f"共 {len(badcase_df)} 条Badcase")

                for _, row in badcase_df.iterrows():
                    with st.expander(f"🔴 [{row['总分']}/5.0] {row.get('conversation_id', '')} — {row.get('agent_name', '')}"):
                        st.markdown(f"**对话内容**")
                        st.text(row["conversation_text"])

                        # 找到低分维度
                        low_dims = []
                        for dim in EVALUATION_DIMENSIONS.keys():
                            if row[dim] <= 2:
                                low_dims.append(f"{dim}({row[dim]}分)")
                        if low_dims:
                            st.error(f"**低分维度**: {'、'.join(low_dims)}")

                        st.info(f"""
                        **改进方向**:
                        - 每句回复前确认是否理解了消费者真实诉求
                        - 即使无法满足诉求，也要表达理解和共情
                        - 说明原因并提供替代方案，而非直接拒绝
                        - 明确告知下一步进展和时间预期
                        """)

        # 导出
        st.divider()
        export_cols = [c for c in display_cols if c in df_results.columns] + ["conversation_text"]
        st.download_button(
            label="⬇️ 下载评估结果CSV",
            data=df_results[export_cols].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name=f"客服质量评估结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
