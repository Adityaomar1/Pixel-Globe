import os
import pptx
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

prs = pptx.Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

blank_layout = prs.slide_layouts[6]

COLOR_BLUE = RGBColor(0, 112, 192)
COLOR_NAVY = RGBColor(11, 25, 44)
COLOR_TEXT = RGBColor(30, 41, 59)
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_LIGHT_BG = RGBColor(248, 250, 252)
COLOR_BORDER = RGBColor(0, 112, 192)
COLOR_PURPLE = RGBColor(107, 33, 168)
COLOR_ORANGE = RGBColor(234, 88, 12)
COLOR_GREEN = RGBColor(22, 101, 52)
COLOR_GREEN_BG = RGBColor(240, 253, 244)

def add_base_decorations(slide, title_text, slide_num):
    # Top Left Oval Badge
    oval = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.5), Inches(0.3), Inches(1.8), Inches(0.7))
    oval.fill.solid()
    oval.fill.fore_color.rgb = RGBColor(250, 245, 255)
    oval.line.color.rgb = COLOR_PURPLE
    oval.line.width = Pt(1.5)
    tf = oval.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = 'YOUR TEAM\nNAME'
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = RGBColor(30, 30, 30)

    # Top Center Title
    tx_box = slide.shapes.add_textbox(Inches(2.5), Inches(0.25), Inches(8.3), Inches(0.8))
    tf = tx_box.text_frame
    p = tf.paragraphs[0]
    p.text = title_text
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = COLOR_NAVY

    # Top Right SIH Label
    tx_sih = slide.shapes.add_textbox(Inches(11.0), Inches(0.25), Inches(1.9), Inches(0.8))
    tf_s = tx_sih.text_frame
    p_s = tf_s.paragraphs[0]
    p_s.text = 'SMART INDIA\nHACKATHON 2026'
    p_s.alignment = PP_ALIGN.RIGHT
    p_s.font.size = Pt(11)
    p_s.font.bold = True
    p_s.font.color.rgb = COLOR_NAVY

    # Bottom Blue Footer Bar
    footer = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.1), Inches(13.333), Inches(0.4))
    footer.fill.solid()
    footer.fill.fore_color.rgb = COLOR_BLUE
    footer.line.fill.background()
    tf_f = footer.text_frame
    p_f = tf_f.paragraphs[0]
    p_f.text = f' @SIH Idea submission- Template                                                                                                {slide_num}'
    p_f.font.size = Pt(11)
    p_f.font.bold = True
    p_f.font.color.rgb = COLOR_WHITE

def create_card_box(slide, left, top, width, height, header_text, bg_color=COLOR_WHITE, border_color=COLOR_BORDER):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = bg_color
    box.line.color.rgb = border_color
    box.line.width = Pt(2)
    
    if header_text:
        tx = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.08), width - Inches(0.2), Inches(0.4))
        tf = tx.text_frame
        p = tf.paragraphs[0]
        p.text = header_text
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_BLUE
    return box

# ==================== SLIDE 1 ====================
slide1 = prs.slides.add_slide(blank_layout)
add_base_decorations(slide1, 'TITLE PAGE', 1)

main_box = create_card_box(slide1, Inches(1.5), Inches(1.3), Inches(10.33), Inches(5.4), '')
tf = main_box.text_frame
p = tf.paragraphs[0]
p.text = '\nCOSMOS // ORBITAL SATELLITE CHANGE DETECTION'
p.alignment = PP_ALIGN.CENTER
p.font.size = Pt(24)
p.font.bold = True
p.font.color.rgb = COLOR_BLUE

p2 = tf.add_paragraph()
p2.text = 'Dual-Model Deep Learning Framework for Bi-Temporal Remote Sensing & Real-Time Geospatial Intelligence\n'
p2.alignment = PP_ALIGN.CENTER
p2.font.size = Pt(13)
p2.font.color.rgb = RGBColor(71, 85, 105)

meta_text = '''
• Problem Statement ID: SIH 1518
• PS Category: Software
• Problem Statement Title: AI-Driven Bi-Temporal Satellite Image Change Detection for Earth Observation & Urban Monitoring
• Theme: Space Technology / Disaster Management / Environment
• Team ID: [Your Team ID]
• Team Name: [Your Team Name]
'''
for line in meta_text.strip().split('\n'):
    p_meta = tf.add_paragraph()
    p_meta.text = line
    p_meta.font.size = Pt(13)
    p_meta.font.bold = True if ':' in line else False
    p_meta.font.color.rgb = COLOR_TEXT

# ==================== SLIDE 2 ====================
slide2 = prs.slides.add_slide(blank_layout)
add_base_decorations(slide2, 'IDEA TITLE & PROPOSED SOLUTION', 2)

create_card_box(slide2, Inches(0.5), Inches(1.2), Inches(6.0), Inches(5.6), 'IDEA / SOLUTION :')
tx_left = slide2.shapes.add_textbox(Inches(0.6), Inches(1.7), Inches(5.8), Inches(5.0))
tf_l = tx_left.text_frame
tf_l.word_wrap = True
p = tf_l.paragraphs[0]
p.text = 'Implementation of an End-to-End Dual-Model Bi-Temporal Deep Learning System for Automated Satellite Change Detection & Real-Time Geospatial Intelligence.'
p.font.size = Pt(11)
p.font.color.rgb = COLOR_TEXT

points_l = [
    'Bi-Temporal Neural Pipeline: Ingests Pre-Event (T1) and Post-Event (T2) satellite imagery to automatically compute sub-pixel structural transformations.',
    'Dual Neural Model Engine: Integrates lightweight attention-based TinyCD (EfficientNet-B0) and differential encoder-decoder Siamese U-Net.',
    'Zero-Downtime Model Switching: Instant dynamic toggling between models in memory without restarting backend servers.',
    'Interactive Space HUD Mission Console: Complete web application featuring draggable Swipe Reveal Slider, sensitivity controls, and real-time telemetry.',
    'Multi-Modal Output Suite: Generates Binary Anomaly Masks, Glowing Holographic Overlays, Continuous Turbo Heatmaps, and 3-Panel Composite Reports.'
]
for pt in points_l:
    p_pt = tf_l.add_paragraph()
    p_pt.text = '◆ ' + pt
    p_pt.font.size = Pt(10.5)
    p_pt.font.color.rgb = COLOR_TEXT

create_card_box(slide2, Inches(6.8), Inches(1.2), Inches(6.0), Inches(2.6), 'Problem Resolution :')
tx_tr = slide2.shapes.add_textbox(Inches(6.9), Inches(1.65), Inches(5.8), Inches(2.0))
tf_tr = tx_tr.text_frame
tf_tr.word_wrap = True
points_tr = [
    'Automated Sub-Second Inference: Replaces tedious, error-prone manual GIS photo-interpretation with instantaneous neural prediction (< 300 ms).',
    'High Environmental Robustness: Eliminates false-positive alerts caused by seasonal vegetation variations, solar illumination shifts, and sensor noise.',
    'Instant Quantitative Telemetry: Automatically calculates changed surface area (4.8%), exact pixel counts (3,200 px), and anomaly severity indices.'
]
p_first = tf_tr.paragraphs[0]
p_first.text = '◆ ' + points_tr[0]
p_first.font.size = Pt(10)
p_first.font.color.rgb = COLOR_TEXT
for pt in points_tr[1:]:
    p_pt = tf_tr.add_paragraph()
    p_pt.text = '◆ ' + pt
    p_pt.font.size = Pt(10)
    p_pt.font.color.rgb = COLOR_TEXT

create_card_box(slide2, Inches(6.8), Inches(4.0), Inches(6.0), Inches(2.8), 'Unique Value Propositions (UVP) :')
tx_br = slide2.shapes.add_textbox(Inches(6.9), Inches(4.45), Inches(5.8), Inches(2.2))
tf_br = tx_br.text_frame
tf_br.word_wrap = True
points_br = [
    'Dynamic Model Switching Matrix: Cross-validate results using both Attention-based and Differential Siamese networks on a single interface.',
    'Interactive Swipe Reveal HUD: Non-technical operators can visually peel and compare satellite layers effortlessly.',
    'Runs with Full Efficiency: Optimized for both local edge devices (CUDA GPU / CPU fallback) and centralized cloud servers.',
    'Zero-Cost Open-Source Stack: Built 100% on PyTorch, OpenCV, and Flask without proprietary GIS licensing fees.'
]
p_first2 = tf_br.paragraphs[0]
p_first2.text = '◆ ' + points_br[0]
p_first2.font.size = Pt(10)
p_first2.font.color.rgb = COLOR_TEXT
for pt in points_br[1:]:
    p_pt = tf_br.add_paragraph()
    p_pt.text = '◆ ' + pt
    p_pt.font.size = Pt(10)
    p_pt.font.color.rgb = COLOR_TEXT

# ==================== SLIDE 3 ====================
slide3 = prs.slides.add_slide(blank_layout)
add_base_decorations(slide3, 'TECHNICAL APPROACH', 3)

create_card_box(slide3, Inches(0.5), Inches(1.2), Inches(6.0), Inches(5.1), 'TECHNOLOGY STACK & ALGORITHMS')
tx_tech = slide3.shapes.add_textbox(Inches(0.6), Inches(1.65), Inches(5.8), Inches(4.5))
tf_t = tx_tech.text_frame
tf_t.word_wrap = True
tech_sections = [
    ('Deep Learning & AI Core:', 'PyTorch 2.7 & Torchvision (CUDA 11.8 Accelerated) — Powers TinyCD (EfficientNet-B0 backbone + Squeeze-and-Excitation attention) and Siamese U-Net (Shared DoubleConv Encoder with differential skip connections).'),
    ('Computer Vision & Preprocessing:', 'OpenCV & Pillow (PIL) — Multi-scale ImageNet & [0,1] adaptive normalization, morphological contour extraction, and continuous Turbo colormap heatmap rendering.'),
    ('Backend & REST APIs:', 'Python 3.10 & Flask 3.1 — High-performance WSGI server with Base64 in-memory streaming REST endpoints (/api/predict, /api/switch_model, /api/info).'),
    ('Frontend HUD & Visualization:', 'HTML5 Canvas, CSS3 Glassmorphism & ES6+ JS — Interactive Starfield engine, telemetry widgets, drag-and-drop stations, and swipe comparison slider.')
]
for idx, (head, body) in enumerate(tech_sections):
    p_h = tf_t.paragraphs[0] if idx == 0 else tf_t.add_paragraph()
    p_h.text = head
    p_h.font.size = Pt(11)
    p_h.font.bold = True
    p_h.font.color.rgb = COLOR_BLUE
    
    p_b = tf_t.add_paragraph()
    p_b.text = body
    p_b.font.size = Pt(10)
    p_b.font.color.rgb = COLOR_TEXT

create_card_box(slide3, Inches(6.8), Inches(1.2), Inches(6.0), Inches(5.1), 'PROCESS FLOW ARCHITECTURE')

node1_a = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(1.7), Inches(2.5), Inches(0.6))
node1_a.fill.solid(); node1_a.fill.fore_color.rgb = RGBColor(224, 242, 254); node1_a.line.color.rgb = COLOR_BLUE
node1_a.text_frame.paragraphs[0].text = '🛰️ T1 Pre-Event'
node1_a.text_frame.paragraphs[0].font.size = Pt(10); node1_a.text_frame.paragraphs[0].font.bold = True
node1_a.text_frame.paragraphs[0].font.color.rgb = COLOR_NAVY

node1_b = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.0), Inches(1.7), Inches(2.5), Inches(0.6))
node1_b.fill.solid(); node1_b.fill.fore_color.rgb = RGBColor(224, 242, 254); node1_b.line.color.rgb = COLOR_BLUE
node1_b.text_frame.paragraphs[0].text = '🛸 T2 Post-Event'
node1_b.text_frame.paragraphs[0].font.size = Pt(10); node1_b.text_frame.paragraphs[0].font.bold = True
node1_b.text_frame.paragraphs[0].font.color.rgb = COLOR_NAVY

node2 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(2.6), Inches(5.4), Inches(0.6))
node2.fill.solid(); node2.fill.fore_color.rgb = COLOR_WHITE; node2.line.color.rgb = COLOR_BORDER
node2.text_frame.paragraphs[0].text = '🔄 Adaptive Preprocessing (256x256 Resizing & Normalization)'
node2.text_frame.paragraphs[0].font.size = Pt(10); node2.text_frame.paragraphs[0].font.bold = True
node2.text_frame.paragraphs[0].font.color.rgb = COLOR_NAVY

node3 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(3.5), Inches(5.4), Inches(0.65))
node3.fill.solid(); node3.fill.fore_color.rgb = RGBColor(255, 247, 237); node3.line.color.rgb = COLOR_ORANGE
node3.text_frame.paragraphs[0].text = '⚡ Neural Inference (CUDA GPU: RTX 2050)\nTinyCD (Attention) ⇄ Siamese U-Net'
node3.text_frame.paragraphs[0].font.size = Pt(10); node3.text_frame.paragraphs[0].font.bold = True
node3.text_frame.paragraphs[0].font.color.rgb = COLOR_ORANGE

node4 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(4.45), Inches(5.4), Inches(0.55))
node4.fill.solid(); node4.fill.fore_color.rgb = COLOR_WHITE; node4.line.color.rgb = COLOR_BORDER
node4.text_frame.paragraphs[0].text = '📊 Sigmoid Probability & Thresholding (0.1 - 0.9)'
node4.text_frame.paragraphs[0].font.size = Pt(10); node4.text_frame.paragraphs[0].font.bold = True
node4.text_frame.paragraphs[0].font.color.rgb = COLOR_NAVY

node5 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(5.3), Inches(5.4), Inches(0.65))
node5.fill.solid(); node5.fill.fore_color.rgb = RGBColor(224, 242, 254); node5.line.color.rgb = COLOR_BLUE
node5.text_frame.paragraphs[0].text = '⚡ Binary Mask   |   🛰️ Holographic Overlay   |   🌌 Turbo Heatmap'
node5.text_frame.paragraphs[0].font.size = Pt(10); node5.text_frame.paragraphs[0].font.bold = True
node5.text_frame.paragraphs[0].font.color.rgb = COLOR_NAVY

status_box = create_card_box(slide3, Inches(0.5), Inches(6.45), Inches(12.3), Inches(0.55), '', bg_color=COLOR_GREEN_BG, border_color=COLOR_GREEN)
tf_sb = status_box.text_frame
p_sb = tf_sb.paragraphs[0]
p_sb.text = '🚀 Product Status: 100% Working Prototype Completed with Live Dual-Model Switching, CUDA GPU Acceleration & Space HUD Web Interface.'
p_sb.alignment = PP_ALIGN.CENTER
p_sb.font.size = Pt(11)
p_sb.font.bold = True
p_sb.font.color.rgb = COLOR_GREEN

# ==================== SLIDE 4 ====================
slide4 = prs.slides.add_slide(blank_layout)
add_base_decorations(slide4, 'FEASIBILITY AND VIABILITY', 4)

col_w = Inches(3.9)
create_card_box(slide4, Inches(0.5), Inches(1.2), col_w, Inches(5.6), 'Feasibility Analysis :')
tx4_1 = slide4.shapes.add_textbox(Inches(0.6), Inches(1.7), col_w - Inches(0.2), Inches(5.0))
tf4_1 = tx4_1.text_frame; tf4_1.word_wrap = True
pts4_1 = [
    'Technical Feasibility: Verified sub-second latency (~298ms - 540ms on NVIDIA RTX 2050 GPU) with lightweight memory footprint (< 55 MB checkpoint size).',
    'Operational Viability: Lightweight web architecture accessible across standard browsers without needing expensive GIS workstations.',
    'Financial Viability: Built 100% on open-source frameworks (PyTorch, Flask, OpenCV) with zero recurring license costs.'
]
for idx, pt in enumerate(pts4_1):
    p = tf4_1.paragraphs[0] if idx == 0 else tf4_1.add_paragraph()
    p.text = '◆ ' + pt; p.font.size = Pt(10.5); p.font.color.rgb = COLOR_TEXT

create_card_box(slide4, Inches(4.7), Inches(1.2), col_w, Inches(5.6), 'Challenges & Risks :')
tx4_2 = slide4.shapes.add_textbox(Inches(4.8), Inches(1.7), col_w - Inches(0.2), Inches(5.0))
tf4_2 = tx4_2.text_frame; tf4_2.word_wrap = True
pts4_2 = [
    'Environmental Noise: Seasonal foliage changes, sun-glint, and cloud shadows can trigger false-positive alarms.',
    'Sensor Discrepancies: Satellite imagery from different sensors/resolutions may cause alignment (co-registration) errors.',
    'Gigapixel Scale: Extremely large geographic orthomosaics require high GPU memory during batch inference.'
]
for idx, pt in enumerate(pts4_2):
    p = tf4_2.paragraphs[0] if idx == 0 else tf4_2.add_paragraph()
    p.text = '◆ ' + pt; p.font.size = Pt(10.5); p.font.color.rgb = COLOR_TEXT

create_card_box(slide4, Inches(8.9), Inches(1.2), col_w, Inches(5.6), 'Mitigation Strategies :')
tx4_3 = slide4.shapes.add_textbox(Inches(9.0), Inches(1.7), col_w - Inches(0.2), Inches(5.0))
tf4_3 = tx4_3.text_frame; tf4_3.word_wrap = True
pts4_3 = [
    'Channel Attention & Diff-Layers: Focuses strictly on structural changes while filtering out seasonal/illumination noise.',
    'Adaptive Preprocessing: Standardized bicubic spatial scaling and model-tailored tensor normalization.',
    'Tiled Sliding-Window Inference: Slices gigapixel satellite rasters into 256x256 tiles with overlapping boundary blending.'
]
for idx, pt in enumerate(pts4_3):
    p = tf4_3.paragraphs[0] if idx == 0 else tf4_3.add_paragraph()
    p.text = '◆ ' + pt; p.font.size = Pt(10.5); p.font.color.rgb = COLOR_TEXT

# ==================== SLIDE 5 ====================
slide5 = prs.slides.add_slide(blank_layout)
add_base_decorations(slide5, 'IMPACT AND BENEFITS', 5)

create_card_box(slide5, Inches(0.5), Inches(1.2), Inches(6.0), Inches(5.6), 'Target Audience Impact :')
tx5_1 = slide5.shapes.add_textbox(Inches(0.6), Inches(1.7), Inches(5.8), Inches(5.0))
tf5_1 = tx5_1.text_frame; tf5_1.word_wrap = True
pts5_1 = [
    'Space Agencies & ISRO/NRSC: High-throughput automated surveillance across multi-temporal satellite constellations.',
    'Disaster Management Authorities (NDRF/SDMA): Instant damage mapping during floods, earthquakes, landslides, and cyclones for prioritized rescue routing.',
    'Urban Planning & Municipal Bodies: Automated tracking of illegal encroachments, unauthorized constructions, and urban sprawl.',
    'Forestry & Environment Departments: Continuous real-time alerts for illegal deforestation, mining activities, and waterbody shrinkage.',
    'Defence & Border Security: Tactical monitoring of remote infrastructure build-up and border line alterations.'
]
for idx, pt in enumerate(pts5_1):
    p = tf5_1.paragraphs[0] if idx == 0 else tf5_1.add_paragraph()
    p.text = '◆ ' + pt; p.font.size = Pt(10.5); p.font.color.rgb = COLOR_TEXT

create_card_box(slide5, Inches(6.8), Inches(1.2), Inches(6.0), Inches(5.6), 'Key Multidimensional Benefits :')
tx5_2 = slide5.shapes.add_textbox(Inches(6.9), Inches(1.7), Inches(5.8), Inches(5.0))
tf5_2 = tx5_2.text_frame; tf5_2.word_wrap = True
benefits = [
    ('🌐 Social Impact:', 'Accelerates humanitarian disaster response times from days to minutes, saving lives and providing transparent rehabilitation assessment.'),
    ('💰 Economic Impact:', 'Delivers 90%+ cost and time reduction over manual field surveys; curbs government revenue losses from illegal land occupation.'),
    ('🌱 Environmental Impact:', 'Enables proactive surveillance of ecologically sensitive zones, national parks, and coastal regulatory zones against illegal degradation.')
]
for idx, (h, b) in enumerate(benefits):
    p_h = tf5_2.paragraphs[0] if idx == 0 else tf5_2.add_paragraph()
    p_h.text = h; p_h.font.size = Pt(12); p_h.font.bold = True; p_h.font.color.rgb = COLOR_BLUE
    p_b = tf5_2.add_paragraph()
    p_b.text = b; p_b.font.size = Pt(10.5); p_b.font.color.rgb = COLOR_TEXT

# ==================== SLIDE 6 ====================
slide6 = prs.slides.add_slide(blank_layout)
add_base_decorations(slide6, 'RESEARCH AND REFERENCES', 6)

create_card_box(slide6, Inches(0.5), Inches(1.2), Inches(6.0), Inches(5.6), 'Primary Research Literature :')
tx6_1 = slide6.shapes.add_textbox(Inches(0.6), Inches(1.7), Inches(5.8), Inches(5.0))
tf6_1 = tx6_1.text_frame; tf6_1.word_wrap = True
pts6_1 = [
    'TinyCD Architecture: Codegoni et al., "TinyCD: A (Not So) Deep Learning Model for Change Detection", IEEE Geoscience and Remote Sensing Letters, 2022.',
    'Siamese U-Net Networks: Daudt et al., "Fully Convolutional Siamese Networks for Change Detection", IEEE International Conference on Image Processing (ICIP), 2018.',
    'Squeeze-and-Excitation Networks: Hu et al., "Squeeze-and-Excitation Networks", IEEE/CVF CVPR.',
    'EfficientNet Backbone: Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML, 2019.'
]
for idx, pt in enumerate(pts6_1):
    p = tf6_1.paragraphs[0] if idx == 0 else tf6_1.add_paragraph()
    p.text = '◆ ' + pt; p.font.size = Pt(10.5); p.font.color.rgb = COLOR_TEXT

create_card_box(slide6, Inches(6.8), Inches(1.2), Inches(6.0), Inches(5.6), 'Datasets & Technical Standards :')
tx6_2 = slide6.shapes.add_textbox(Inches(6.9), Inches(1.7), Inches(5.8), Inches(5.0))
tf6_2 = tx6_2.text_frame; tf6_2.word_wrap = True
pts6_2 = [
    'SYSU-CD Dataset: Sun Yat-sen University Bi-Temporal Remote Sensing Benchmark for urban and land-cover change detection.',
    'LEVIR-CD Dataset: Large-Scale Building Change Detection Dataset with high-resolution aerial imagery pairs.',
    'Deep Learning Framework: PyTorch AI & Torchvision Documentation (https://pytorch.org).',
    'Computer Vision Standards: OpenCV 4.x Image Processing Library (https://opencv.org).',
    'Web Microservices: Flask WSGI Framework & REST API Specifications (https://flask.palletsprojects.com).'
]
for idx, pt in enumerate(pts6_2):
    p = tf6_2.paragraphs[0] if idx == 0 else tf6_2.add_paragraph()
    p.text = '◆ ' + pt; p.font.size = Pt(10.5); p.font.color.rgb = COLOR_TEXT

# Save PPTX
pptx_path = 'C:/Users/legal/OneDrive/Desktop/SIH_2026_Presentation.pptx'
prs.save(pptx_path)
print('SUCCESS: Native PowerPoint PPTX generated at:', pptx_path)
