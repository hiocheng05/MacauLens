import gradio as gr
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# ================= 1. 基础配置 =================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('设备:', device)
if device == 'cuda':
    print('GPU 型号:', torch.cuda.get_device_name(0))

# 类别映射字典：因为 ImageFolder 是按文件夹拼音/英文排序打标签的，
# 所以这里建立英文与中文的映射，用于网页显示。
CLASS_MAP = {
    "barra": "妈阁庙",
    "eiffel_tower": "澳门巴黎铁塔",
    "lighthouse": "东望洋灯塔",
    "macau_tower": "澳门旅游塔",
    "saopaulo": "大三巴牌坊"
}
# 将字典的键提取成列表，方便根据索引位置获取对应的英文标签
class_keys = list(CLASS_MAP.keys())

# 导览信息库：这是本作品“交互应用”的核心特色。
# 模型负责识别建筑，识别结果通过字典映射，自动生成对应的文化看点与路线。
GUIDE = {
    "barra": {          "note": "澳门最古老的庙宇之一。Macau的名字也是由此而来。", 
                        "walk": "附近有港务局大楼、郑家大屋，非常适合串成一条世遗历史步行线。"},
    "eiffel_tower": {   "note": "按巴黎埃菲尔铁塔二分之一比例建造。晚上有绚丽的灯光秀，极具浪漫气息。", 
                        "walk": "位于金沙度假区，周边都是欧式建筑群，隔壁草坪机位最出片。"},
    "lighthouse": {     "note": "中国海岸线上第一座现代灯塔，建于1865年。矗立在东望洋山山顶。", 
                        "walk": "旁边就是东望洋炮台，还可以去+二龙喉公园坐松山缆车。"},
    "macau_tower": {    "note": "澳门的现代地标，338米高。它是全球知名的蹦极胜地，也是俯瞰澳门全景的最佳视角。", 
                        "walk": "塔下有大型商场，旁边的西湾湖畔非常适合傍晚散步看夕阳。"},
    "saopaulo": {       "note": "这是圣保禄大教堂火灾后仅存的前壁。巴洛克风格的立面上融合了中文字和菊花图案。", 
                        "walk": "顺着石阶往下走，途经恋爱巷，再走 5 分钟就能到达议事亭前地。"},
}

# ================= 2. 模型加载机制 =================
# 定义一个通用函数，用来加载我们之前保存的 4 个模型
def load_model(weight_path):
    # 1. 先搭建一个结构相同的“空壳子”模型（不加载默认权重）
    m = models.resnet18(weights=None)
    # 2. 修改输出层为 5 分类
    m.fc = nn.Linear(m.fc.in_features, 5)
    # 3. 把训练好的参数（权重）装入空壳子中，map_location 确保能自动适配 CPU/GPU
    m.load_state_dict(torch.load(weight_path, map_location=device))
    # 4. 放到对应设备上，并开启评估模式（关闭 Dropout 等，确保预测稳定）
    m = m.to(device)
    m.eval()
    return m

print("正在加载四个模型...")
# 从 Train_test 文件夹中分别加载 4 种策略的模型
model_a = load_model('./Train_test/model_a.pth')
model_b = load_model('./Train_test/model_b.pth')
model_c = load_model('./Train_test/model_c.pth')
model_d = load_model('./Train_test/model_d.pth')
print("模型加载完毕！")

# 图片预处理流水线：网页传进来的图片，必须经过和训练集验证集一模一样的处理
transform = transforms.Compose([
    transforms.Resize(256),            # 短边缩放
    transforms.CenterCrop(224),        # 中心裁剪
    transforms.ToTensor(),             # 转张量 (0-1)
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 标准化
])

# ================= 3. 核心推理逻辑 =================
# 这个函数是网页点击按钮后执行的核心代码
def predict_and_guide(img):
    # 容错处理：如果用户没传图片就点按钮
    if img is None:
        return {}, "请先上传一张照片哦！", "", ""
    
    # 1. 图片预处理：加上 unsqueeze(0) 增加 batch 维度，变成 [1, 3, 224, 224]
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 2. 关闭梯度计算，让四个模型同时对这张图进行推理（前向传播）
    with torch.no_grad():
        out_a = model_a(img_tensor)
        out_b = model_b(img_tensor)
        out_c = model_c(img_tensor)
        out_d = model_d(img_tensor)
        
    # 3. 提取 A 策略（主模型）的结果用于展示和导览
    probs_a = torch.softmax(out_a, dim=1)[0]  # 计算概率
    top3_prob_a, top3_idx_a = torch.topk(probs_a, 3) # 取概率最高的前 3 名
    
    # 将 Top-3 拼装成字典，方便 Gradio 的 Label 组件自动渲染成进度条
    top3_dict = {}
    for i in range(3):
        en_name = class_keys[top3_idx_a[i].item()]
        top3_dict[CLASS_MAP[en_name]] = float(top3_prob_a[i].item())
        
    top1_en_a = class_keys[top3_idx_a[0].item()] # 第 1 名的英文标签
    top1_prob_a = top3_prob_a[0].item()          # 第 1 名的置信度
    
    # 4. 查字典：根据第 1 名，去 GUIDE 字典里找对应的看点与漫游路线
    guide = GUIDE[top1_en_a]
    note_text = guide["note"]
    walk_text = guide["walk"]
    
    # 5. 【交互亮点】低置信度兜底机制：如果模型自己都拿不准（低于 65%）
    if top1_prob_a < 0.65:
        note_text = f"⚠️ 主模型不太确定（置信度仅 {top1_prob_a*100:.1f}%）。\n如果您拍的是局部特写，建议换一张全景图。\n系统猜测：{note_text}"
        walk_text = "请确认地标后再查看推荐路线。"

    # 6. 提取 A, B, C, D 策略的预测结果（只取 Top-1）用于底部的折叠面板对比
    def get_top1(probs):
        idx = torch.argmax(probs).item()
        conf = probs[idx].item()
        return f"{CLASS_MAP[class_keys[idx]]} ({conf*100:.1f}%)"

    # 分别算出四个模型的预测文本
    res_a = get_top1(torch.softmax(out_a, dim=1)[0])
    res_b = get_top1(torch.softmax(out_b, dim=1)[0])
    res_c = get_top1(torch.softmax(out_c, dim=1)[0])
    res_d = get_top1(torch.softmax(out_d, dim=1)[0])
    
    # 把四个策略的结果拼成一段多行文本
    other_ai_text = (
        f"策略 A (冻结+SGD): {res_a}\n\n"
        f"策略 B (冻结+Adam): {res_b}\n\n"
        f"策略 C (微调+SGD): {res_c}\n\n"
        f"策略 D (微调+Adam): {res_d}"
    )

    # 返回给网页前端渲染：Top3字典、看点文本、漫游文本、对比文本
    return top3_dict, note_text, walk_text, other_ai_text

# ================= 4. 搭建网页 UI =================
# 使用 Blocks 框架，类似搭积木，支持复杂的排版
with gr.Blocks(title="MacauLens", theme=gr.themes.Soft()) as demo:
    # 网页标题和简介
    gr.Markdown("# 🏛️ MacauLens · 澳门城市地标识别与导览助手")
    gr.Markdown("上传照片，主 AI 模型将为您识别地标并规划路线。您可以展开下方面板，查看其他 AI 策略的对比意见。")
    
    # 横向布局 (Row)：分为左列（输入）和右列（输出）
    with gr.Row():
        # 左侧：输入区 (Column 占 1 份宽度)
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="📸 请上传照片", height=445)
            btn = gr.Button("🚀 开始识别与导览", variant="primary", size="lg")
            
        # 右侧：输出区 (Column 占 1 份宽度)
        with gr.Column(scale=1):
            # 1. 主模型结果展示 (带百分比的进度条)
            gr.Markdown("### 🎯 主模型识别结果 (策略 A)")
            lbl_result = gr.Label(show_label=False, num_top_classes=3)
            
            # 2. 导览信息卡片
            with gr.Group():
                gr.Markdown("### 📖 建筑身份看点")
                txt_note = gr.Textbox(show_label=False, lines=3)
            with gr.Group():
                gr.Markdown("### 🚶 周边顺路漫游")
                txt_walk = gr.Textbox(show_label=False, lines=2)

    # 3. 折叠面板：展示其他三个模型的观点
    with gr.Accordion("👀 展开查看其他 AI 策略的预测对比", open=False):
        # 按你的要求修改了文本
        gr.Markdown("为了验证系统稳定性，在此内置了四种训练策略。以下是其他模型的判断结果( A / D 更为可信 )：")
        txt_other_ai = gr.Textbox(show_label=False, lines=7) # 高度拉高一点，容纳 4 行文本

    # 事件绑定：点击 btn 按钮 -> 调用 predict_and_guide 函数 -> 把结果分别塞进四个输出框里
    btn.click(fn=predict_and_guide, inputs=img_input, outputs=[lbl_result, txt_note, txt_walk, txt_other_ai])

# 启动服务器
if __name__ == "__main__":
    # 从 Render 的环境变量中读取端口，如果本地运行则默认使用 7860
    port = int(os.environ.get("PORT", 7860))
    # 关键：绑定到 0.0.0.0，并指定服务器端口
    demo.launch(server_name="0.0.0.0", server_port=port)