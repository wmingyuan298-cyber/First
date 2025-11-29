import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
try:
    import shap  # optional; we will gracefully fallback if not present
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# -----------------------------
# Config - Adjustable Font Sizes
# -----------------------------
# 说明：
# - 保持页面原有风格、内容、默认字体大小完全不变
# - 只需修改下方 UI_CSS_FONT_SIZES 和 PLOT_FONT_SIZES 中的值即可全局调整对应文字大小
# - 单位：CSS 使用字符串单位（px/em/rem 等），Matplotlib 使用数字（字号）
# 
# 快速对照表（改哪个就变哪段文字）：
# A. 页面主标题（页面最上方大标题“🔬 Black Phosphorus Gas Response Predictor”）
#    -> UI_CSS_FONT_SIZES['main_title']
# B. 预测结果横幅（彩色大横幅）
#    -> 标题字号：UI_CSS_FONT_SIZES['result_banner_title']
#    -> 横幅内第二行细节（Weighted Score/Threshold 或 Gas 来源）：UI_CSS_FONT_SIZES['result_banner_detail']
# C. 侧边栏标题（“⚙️ Input Parameters”）
#    -> UI_CSS_FONT_SIZES['sidebar_h2_title']
# D. 侧边栏五个特征名称标签（Eads/ℇp/VBM/CBM/Ef）
#    -> UI_CSS_FONT_SIZES['sidebar_feature_label']
# E. 侧边栏输入框容器与标签（影响容器中文字，如单位提示）
#    -> 容器字号：UI_CSS_FONT_SIZES['sidebar_input_container']
#    -> 标签字号：UI_CSS_FONT_SIZES['sidebar_input_label']
# F. 侧边栏数字输入框中的数值（输入框里显示的数字大小）
#    -> UI_CSS_FONT_SIZES['sidebar_number_input_value']
# G. 阈值滑块（Decision Threshold）
#    -> 滑块标题“🎯 Decision Threshold”：UI_CSS_FONT_SIZES['slider_label']
#    -> 滑块当前值（显示在滑块上方的数字）：UI_CSS_FONT_SIZES['slider_current_value']
#    -> 滑块两端范围数字（0.00 / 1.00）：UI_CSS_FONT_SIZES['slider_range']
#    -> 范围数字内部文字（兼容性补充）：UI_CSS_FONT_SIZES['slider_range_inner']
# H. 侧边栏 Predict 按钮文字
#    -> UI_CSS_FONT_SIZES['predict_button_text']
# I. 页面各小节标题（h3，如“📋 Feature Decision Details”“📊 …”“📈 …”）
#    -> UI_CSS_FONT_SIZES['section_title_h3']
# J. Top-5 特征的徽章（紫色圆角标签）
#    -> UI_CSS_FONT_SIZES['feature_badge']
# K. “Top-5 Features:” 标签文字
#    -> UI_CSS_FONT_SIZES['top5_label']
# L. 表格（st.table）整体字号与与下方内容间距
#    -> 字号：UI_CSS_FONT_SIZES['table_font']
#    -> 表格底部间距：UI_CSS_FONT_SIZES['table_bottom_margin']
# 
# 图表（Matplotlib）字号对照：
# - 全局：PLOT_FONT_SIZES['axes_title'/'axes_label'/'tick_label'/'legend']
# - Ranges 图（“Feature Value Ranges Visualization”）：
#     标题：ranges_title；x轴标签：ranges_xlabel；y轴特征刻度：ranges_ytick
# - 加权分瀑布图（“Weighted Score Sum Calculation …”）：
#     标题：waterfall_title；x轴：waterfall_xlabel；y轴特征刻度：waterfall_ytick；
#     决策标签“Decision: …”：waterfall_decision；
#     底部公式说明：waterfall_formula；
#     每个小条上的数值：annotation_small
UI_CSS_FONT_SIZES = {
    # 顶部主标题 "🔬 Black Phosphorus Gas Response Predictor"
    'main_title': '2.5rem',  # 页面最上方大标题（"🔬 Black Phosphorus Gas Response Predictor"）

    # 预测结果横幅
    'result_banner_title': '32px',  # 预测结果横幅主标题（大字）
    'result_banner_detail': '2em',  # 横幅第二行细节文本（Weighted Score / Threshold 或 Gas 来源）- HTML中实际使用2em

    # 侧边栏
    'sidebar_h2_title': '2.0em',  # 侧边栏大标题 "⚙️ Input Parameters" 的字号
    'sidebar_feature_label': '1.4em',  # 侧边栏特征名称标签（Eads/ℇp/VBM/CBM/Ef）- 与 Threshold 和 Predict 统一
    'feature_inline_label': '2.8em',  # 行内特征标签（与输入框同一行显示的特征名称）
    'sidebar_input_container': '1.2em',  # 侧边栏输入框容器内通用文字（如占位、单位）
    'sidebar_input_label': '1.2em',  # 侧边栏输入框标签文本（在输入框上方/左侧的字）
    'sidebar_number_input_value': 'calc(1.6em + 8px)',  # 侧边栏数字输入框中的数字（输入框里显示的值）- 在1.4em基础上再增大8px

    # 滑块（阈值）- 与五个特征和 Predict 按钮统一字体大小
    'slider_label': '1.5em',        # 阈值滑块标题（"Threshold" 这行字）- 与特征标签统一
    'slider_current_value': '1.5em', # 滑块当前值（显示在滑块上方的数字）- 增大字体
    'slider_range': '1.5em',        # 滑块两端范围数字（0.00 / 1.00）- 增大字体
    'slider_range_inner': '1.5em',  # 滑块范围数字内部元素（兼容不同 DOM 结构，用于确保范围字号生效）

    # 按钮（Predict）- 与五个特征和 Threshold 统一字体大小
    'predict_button_text': '1.4em',  # 侧边栏 "Predict" 按钮文字 - 与特征标签统一

    # 章节标题（h3）
    'section_title_h3': '2.5em',  # 页面各小节标题（h3），如"📋 …""📊 …""📈 …"

    # 特征徽章（Top-5）
    'feature_badge': '1.4em',  # Top-5 特征徽章（紫色圆角标签）

    # Top-5 Features 标签
    'top5_label': '2.0em',  # "Top-5 Features:" 标签文字

    # 表格
    'table_font': '36px',   # 最终覆盖表格的字号（页面里会再次注入 CSS 覆盖）- 进一步增大字体
    'table_bottom_margin': '7rem',
}

# Matplotlib 字体大小（用于图表内文字）
PLOT_FONT_SIZES = {
    # 全局 rcParams - 统一字体大小为 32（与 h3 标题一致）
    'axes_title': 32,   # 全局默认轴标题字号（rcParams: axes.titlesize）
    'axes_label': 32,   # 全局默认坐标轴标签字号（与标题统一）
    'tick_label': 32,   # 坐标轴刻度标签（与标题统一）
    'legend': 32,       # 图例字体（与标题统一）

    # Ranges 图 - 统一字体大小为 32（与标题一致）
    'ranges_title': 32,
    'ranges_xlabel': 32,
    'ranges_ytick': 32,

    # Weighted Score Waterfall 图 - 统一字体大小为 32（与 h3 标题 "Weighted Score Sum Calculation Visualization" 一致，2.5em ≈ 32pt）
    'waterfall_title': 32,
    'waterfall_xlabel': 32,  # 与标题一致
    'waterfall_ytick': 32,   # 与标题一致
    'waterfall_decision': 32,  # 与标题一致
    'waterfall_formula': 32,  # 与标题一致
    'annotation_small': 32,  # 条形上的数值，与标题一致
}


# -----------------------------
# Data/Features
# -----------------------------
DATA_FILE = os.path.join(os.path.dirname(__file__), 'BP-qiti.xlsx')
FEATURES = ['Eads', 'd', '∆Q', 'ℇp', 'VBM', 'CBM', 'Ef', 'Eg', 'WF']
TARGET = 'Response'

# ⭐ 嵌入的气体响应数据字典（从Excel提取，不依赖Excel文件）⭐
GAS_RESPONSE_DATA = {
    'NO2': 1,
    'N2': 0,
    'NO': 1,
    'H2': 0,
    'CO': 0,
    'NH3': 1,
    'CH3CH2COH': 1,
    'C3H9O3P': 1,
    'C6H6': 0,
    'CH3OH': 1,
    'C7H8': 0,
    'CH3COH': 0,
    'C3H9N': 1,
    'CCl3H': 0,
    'CCl2H2': 0,
    'CH3CHOHCH3': 0,
    'CO2': 0,
    'H2O': 1,
    'H2S': 1,
    'CH3COCH3': 1,
    'CH3CH2OH': 0
}


def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE)
    return df.copy()


def compute_importances(df: pd.DataFrame) -> pd.Series:
    X = df[FEATURES]
    y = df[TARGET]
    pipe = make_pipeline(StandardScaler(), ExtraTreesClassifier(n_estimators=400, random_state=42))
    pipe.fit(X, y)
    if HAS_SHAP:
        try:
            explainer = shap.Explainer(pipe.named_steps['extratreesclassifier'])
            shap_vals = explainer(X)
            vals = shap_vals.values
            if vals.ndim == 3:
                vals = vals[..., 1]
            mean_abs = np.abs(vals).mean(axis=0)
            imp = pd.Series(mean_abs, index=FEATURES)
            return imp / imp.sum()
        except Exception:
            pass
    imp = pd.Series(pipe.named_steps['extratreesclassifier'].feature_importances_, index=FEATURES)
    return imp / imp.sum()


def build_summary_df(df: pd.DataFrame, importances: pd.Series) -> pd.DataFrame:
    res = []
    df_res = df[df[TARGET] == 1]
    df_non = df[df[TARGET] == 0]
    for feat in FEATURES:
        if feat not in df.columns:
            continue
        res.append({
            'Feature': feat,
            'Importance': float(importances.get(feat, 0.0)),
            'Response_min': float(df_res[feat].min()),
            'Response_max': float(df_res[feat].max()),
            'NonResponse_min': float(df_non[feat].min()),
            'NonResponse_max': float(df_non[feat].max()),
        })
    s = pd.DataFrame(res)
    s = s[s['Importance'] > 0].reset_index(drop=True)
    return s


def predict_response_weighted(new_sample: dict, summary_df: pd.DataFrame, threshold: float = 0.5):
    weighted_scores = []
    details = []
    eps = 1e-9
    total_importance = summary_df['Importance'].sum()
    for _, row in summary_df.iterrows():
        feat = row['Feature']
        val = new_sample.get(feat, None)
        if val is None:
            continue
        imp = row['Importance'] / total_importance if total_importance > 0 else 0.0
        rmin, rmax = row['Response_min'], row['Response_max']
        nmin, nmax = row['NonResponse_min'], row['NonResponse_max']
        rcenter = (rmin + rmax) / 2
        ncenter = (nmin + nmax) / 2
        d_resp = abs(val - rcenter)
        d_non = abs(val - ncenter)
        score = (d_non + eps) / (d_resp + d_non + eps)
        weighted_score = score * imp
        weighted_scores.append(weighted_score)
        details.append({
            'Feature': feat,
            'Value': val,
            'Importance': float(imp),
            'Response_center': rcenter,
            'NonResponse_center': ncenter,
            'Distance_to_Response': float(d_resp),
            'Distance_to_NonResponse': float(d_non),
            'Score': float(score),
            'WeightedScore': float(weighted_score),
        })
    details_df = pd.DataFrame(details)
    total_weighted_score = sum(weighted_scores) if weighted_scores else 0.0
    result = 'Response' if total_weighted_score >= threshold else 'Non-Response'
    return result, total_weighted_score, details_df


def plot_ranges(summary_df: pd.DataFrame, new_sample: dict):
    # 增大图表尺寸，为 32pt 的大字体留出充足空间
    fig, ax = plt.subplots(figsize=(18, 12), facecolor='white')
    ax.set_facecolor('#f8f9fa')
    
    # 增加特征之间的间距，避免重叠
    feature_spacing = 1.5  # 从默认的 1.0 增加到 1.5
    y_positions = np.arange(len(summary_df)) * feature_spacing
    
    for i, row in summary_df.iterrows():
        feat = row['Feature']
        y = y_positions[i]
        # 增大线条宽度以适应更大的图表
        ax.plot([row['Response_min'], row['Response_max']], [y, y], 
                color='#38ef7d', linewidth=12, alpha=0.8, solid_capstyle='round',
                label='Response range' if i == 0 else "", zorder=1)
        ax.plot([row['NonResponse_min'], row['NonResponse_max']], [y, y], 
                color='#667eea', linewidth=12, alpha=0.8, solid_capstyle='round',
                label='Non-response range' if i == 0 else "", zorder=1)
        if feat in new_sample:
            # 增大散点大小以适应更大的图表
            ax.scatter(new_sample[feat], y, color='#f39c12', s=300, edgecolor='white', 
                      linewidth=4, zorder=10, marker='o', label='Current value' if i == 0 else "")
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary_df['Feature'], fontsize=PLOT_FONT_SIZES['ranges_ytick'], fontweight='600')
    
    # 增大 X 轴标签间距
    ax.set_xlabel('Feature Value', fontsize=PLOT_FONT_SIZES['ranges_xlabel'], 
                  fontweight='bold', color='#2c3e50', labelpad=20)
    
    # 标题使用大字体
    ax.set_title('Feature Value Ranges Visualization', fontsize=PLOT_FONT_SIZES['ranges_title'], 
                 fontweight='bold', color='#2c3e50', pad=30)
    
    # 图例使用大字体，放在右上角，去掉背景填充
    legend = ax.legend(fontsize=PLOT_FONT_SIZES['legend'], loc='upper right', 
                      frameon=False,  # 去掉边框和填充
                      borderpad=2.0, labelspacing=1.5,
                      bbox_to_anchor=(1.0, 1.0))  # 精确定位到右上角（使用 1.0, 1.0 确保在最右上角）
    # 增大图例中的线条和标记大小
    for line in legend.get_lines():
        line.set_linewidth(8)
        line.set_markersize(20)
    
    # 增大坐标轴刻度字体
    ax.tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZES['tick_label'])
    
    ax.grid(True, alpha=0.2, axis='x', linestyle='--', linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dee2e6')
    ax.spines['bottom'].set_color('#dee2e6')
    
    # 调整 Y 轴范围，为顶部和底部留出更多空间
    y_min = -0.5
    y_max = y_positions[-1] + feature_spacing * 0.8 if len(y_positions) > 0 else 1
    ax.set_ylim(y_min, y_max)
    
    # 调整布局，为更大的字体和图例留出充足空间
    # 右侧留出更多空间给图例
    fig.tight_layout(rect=[0, 0.05, 0.95, 0.98])  # 右侧留出 5% 的空间给图例
    return fig


def plot_weighted_score_waterfall(details_df: pd.DataFrame, total_score: float, threshold: float):
    # 进一步增大图表尺寸，为 32pt 的大字体留出充足空间
    fig, ax = plt.subplots(figsize=(22, 16), facecolor='white')
    ax.set_facecolor('#f8f9fa')
    df_sorted = details_df.copy()
    df_sorted['AbsWeightedScore'] = df_sorted['WeightedScore'].abs()
    df_sorted = df_sorted.sort_values('AbsWeightedScore', ascending=True)
    cumulative = 0.0
    # 进一步增加条形间距，确保大字体重叠，并为图例留出充足空间
    bar_spacing = 3.5  # 从 3.0 增加到 3.5
    y_pos = np.arange(len(df_sorted)) * bar_spacing
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_sorted)))
    
    for i, row in enumerate(df_sorted.itertuples()):
        weighted_score = row.WeightedScore
        color = colors[i]
        # 进一步增大条形高度，为大字体留出空间
        bar_height = 1.8  # 从 1.5 增加到 1.8
        ax.barh(y_pos[i], weighted_score, left=cumulative, color=color, alpha=0.8, 
                edgecolor='white', linewidth=4, height=bar_height)
        label_x = cumulative + weighted_score / 2
        # 条形上的数值使用大字体
        # 根据条形宽度决定显示位置，确保清晰可见
        if weighted_score < 0.015:
            # 非常小的数值显示在条形右侧外部，避免被遮挡
            ax.text(cumulative + weighted_score + 0.015, y_pos[i], f'{weighted_score:.4f}', 
                    ha='left', va='center', fontsize=PLOT_FONT_SIZES['annotation_small'], fontweight='bold',
                    color='#2c3e50', zorder=50, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                    edgecolor='none', alpha=0.8))
        elif weighted_score < 0.03:
            # 较小的数值显示在条形右侧外部
            ax.text(cumulative + weighted_score + 0.01, y_pos[i], f'{weighted_score:.4f}', 
                    ha='left', va='center', fontsize=PLOT_FONT_SIZES['annotation_small'], fontweight='bold',
                    color='#2c3e50', zorder=50)
        else:
            # 大数值显示在条形内部，使用对比色
            ax.text(label_x, y_pos[i], f'{weighted_score:.4f}', 
                    ha='center', va='center', fontsize=PLOT_FONT_SIZES['annotation_small'], fontweight='bold',
                    color='white' if weighted_score > 0.05 else '#2c3e50',
                    zorder=50)
        cumulative += weighted_score
    
    # 增大参考线的线宽
    ax.axvline(threshold, color='#f39c12', linestyle='--', linewidth=4, alpha=0.9,
               label=f'Threshold: {threshold:.3f}', zorder=10)
    ax.axvline(total_score, color='#667eea', linestyle='-', linewidth=4, alpha=0.9,
               label=f'Total Score: {total_score:.4f}', zorder=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['Feature'], fontsize=PLOT_FONT_SIZES['waterfall_ytick'], fontweight='600')
    # 增大 X 轴标签间距
    ax.set_xlabel('Cumulative Weighted Score', fontsize=PLOT_FONT_SIZES['waterfall_xlabel'], 
                  fontweight='bold', color='#2c3e50', labelpad=20)
    
    # 标题使用大字体
    ax.set_title('Weighted Score Sum Calculation\n(Score × Weight for each feature)', 
                 fontsize=PLOT_FONT_SIZES['waterfall_title'], fontweight='bold', color='#2c3e50', pad=35)
    
    decision = 'Response' if total_score >= threshold else 'Non-Response'
    decision_color = '#38ef7d' if decision == 'Response' else '#667eea'
    
    # 决策标签使用大字体，去掉蓝色外框和背景填充
    decision_x = 0.95  # Decision 的 x 位置（右对齐）
    decision_y = 0.92  # Decision 的 y 位置
    
    fig.text(decision_x, decision_y, f'Decision: {decision}', 
             ha='right', va='top', fontsize=PLOT_FONT_SIZES['waterfall_decision'], fontweight='bold', color=decision_color,
             # 去掉背景填充：不设置 bbox 或设置 facecolor='none'
             transform=fig.transFigure)
    
    # 图例左端与 Decision 左端对齐
    # Decision 右对齐在 0.95，为了让图例左端对齐，使用 upper left 定位
    # bbox_to_anchor 的 x 坐标设置为 Decision 的左端位置（估算：0.95 - 0.15 ≈ 0.80）
    # 更精确的方法是使用相同的 x 坐标，但让图例左对齐
    legend_x = decision_x - 0.15  # 估算 Decision 文字宽度，让图例左端对齐
    legend = ax.legend(fontsize=PLOT_FONT_SIZES['waterfall_decision'], loc='upper left', 
                      frameon=False,  # 去掉边框和填充
                      borderpad=2.0, labelspacing=1.5,
                      bbox_to_anchor=(legend_x, decision_y), ncol=1)  # 图例左端对齐
    # 增大图例中的线条和标记大小
    for line in legend.get_lines():
        line.set_linewidth(5)
        line.set_markersize(18)
    
    ax.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dee2e6')
    ax.spines['bottom'].set_color('#dee2e6')
    
    # 增大坐标轴刻度字体
    ax.tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZES['tick_label'])
    
    # 进一步增大 X 轴范围，为右侧的数值标签和图例留出充足空间
    x_max = max(total_score, threshold) * 1.4  # 从 1.35 增加到 1.4
    ax.set_xlim(0, x_max)
    
    # 进一步调整 y 轴范围，为底部公式和顶部留出更多空间
    y_min = -4.5  # 从 -4.0 增加到 -4.5，为底部公式留出更多空间
    y_max = y_pos[-1] + bar_spacing * 1.2 if len(y_pos) > 0 else 1  # 增加顶部空间
    ax.set_ylim(y_min, y_max)
    
    # 调整公式位置，避免与坐标轴标签重叠，使用大字体，去掉蓝色外框和背景填充
    ax.text(x_max * 0.45, -3.5,  # 从 -3.2 调整到 -3.5，增加与坐标轴的距离
            f'Formula: Weighted Score = Score × Feature Weight\n'
            f'Total = Σ(Weighted Score) = {total_score:.4f}',
            fontsize=PLOT_FONT_SIZES['waterfall_formula'], style='italic', color='#34495e',
            # 去掉背景填充：不设置 bbox
            zorder=100)
    
    # 调整布局，为更大的字体、图例和决策标签留出充足空间
    # 顶部和右侧留出更多空间
    fig.tight_layout(rect=[0, 0.1, 0.95, 0.95])  # 底部 10%，顶部 5%，右侧 5% 的空间
    return fig


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(
    page_title='BP Response Predictor', 
    layout='wide',
    page_icon='🔬',
    initial_sidebar_state='expanded'
)

# Global matplotlib styles
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': PLOT_FONT_SIZES['axes_title'],
    'axes.labelsize': PLOT_FONT_SIZES['axes_label'],
    'xtick.labelsize': PLOT_FONT_SIZES['tick_label'],
    'ytick.labelsize': PLOT_FONT_SIZES['tick_label'],
    'legend.fontsize': PLOT_FONT_SIZES['legend'],
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
})

# 原有完整样式（不改动，保证默认外观一致）
st.markdown(
    """
    <style>
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'DejaVu Sans', sans-serif;
    }
    
  /* ============================================
      隐藏左上角的关闭按钮（X按钮）
      ============================================ */
  /* 根据实际HTML结构精确隐藏关闭按钮（X按钮） */
  /* 关闭按钮的实际属性：kind="header", data-testid="baseButton-header" */
  /* 方法1: 通过 data-testid 精确匹配关闭按钮 */
  [data-testid="stSidebar"] button[data-testid="baseButton-header"],
  [data-testid="stSidebar"] button[kind="header"],
  [data-testid="stSidebarContent"] button[data-testid="baseButton-header"],
  [data-testid="stSidebarContent"] button[kind="header"],
  /* 方法2: 通过类名匹配（如果上面的方法不够精确） */
  [data-testid="stSidebar"] button.st-emotion-cache-ztfqz8,
  [data-testid="stSidebarContent"] button.st-emotion-cache-ztfqz8 {
      display: none !important;
      visibility: hidden !important;
  }
  
  /* ============================================
       主标题 "🔬 Black Phosphorus Gas Response Predictor" 的字体大小设置
       ============================================ */
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* 预测结果横幅 */
    .result-banner {
        padding: 20px 30px;
        border-radius: 12px;
        font-size: 32px;
        font-weight: 700;
        color: #ffffff;
        display: inline-block;
        margin: 15px 0 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        width: 100%;
        text-align: center;
    }
    .result-banner > div { font-size: 0.9em !important; }
    .result-banner:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.3); }
    .result-green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .result-gray { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 0 1rem 0.3rem 1rem !important;
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] > div:first-child { margin-top: 0 !important; padding-top: 0 !important; }
    [data-testid="stSidebarContent"] { padding-top: 0.2rem !important; margin-top: 0 !important; }
    [data-testid="stSidebar"] > *:first-child { margin-top: 0 !important; padding-top: 0 !important; }
    [data-testid="stSidebar"] { position: relative; top: 0 !important; }
    [data-testid="stSidebar"] .st-emotion-cache-16txtl3,
    [data-testid="stSidebar"] [class*="st-emotion-cache-16txtl3"],
    [data-testid="stSidebar"] [class*="eczjsme4"],
    [data-testid="stSidebar"] div.st-emotion-cache-16txtl3.eczjsme4 { margin-top: 0 !important; padding-top: 0 !important; }
    [data-testid="stSidebar"] button.st-emotion-cache-ztfqz8,
    [data-testid="stSidebar"] [class*="st-emotion-cache-ztfqz8"] { margin-top: 0 !important; padding-top: 0 !important; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] { margin-top: 0 !important; padding-top: 0 !important; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { margin-top: 0 !important; padding-top: 0 !important; }
    [data-testid="stSidebar"] .element-container,
    [data-testid="stSidebar"] [class*="element-container"] { margin-top: 0 !important; padding-top: 0 !important; }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] [class*="stMarkdown"] { margin-top: 0 !important; padding-top: 0 !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { margin-top: 0 !important; padding-top: 0 !important; }
    [data-testid="stSidebar"] .st-emotion-cache-eqffof,
    [data-testid="stSidebar"] [class*="st-emotion-cache-eqffof"] { margin-top: 0 !important; padding-top: 0 !important; }
    [data-testid="stSidebar"] > *:first-child,
    [data-testid="stSidebar"] > div:first-child > *:first-child { margin-top: 0 !important; padding-top: 0 !important; }

    /* 侧边栏标题 */
    [data-testid="stSidebar"] h2 {
        font-size: 2.0em !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
        padding: 0.1rem 0 0.2rem 0 !important;
        margin: 0.1rem 0 0.3rem 0 !important;
        border-bottom: 2px solid #667eea;
    }
    
    [data-testid="stSidebar"] h3 { font-size: 1.1em !important; margin: 0.3rem 0 !important; padding-bottom: 0.2rem !important; }

    /* 侧边栏五个特征标签 - 字体大小由 CSS 变量控制，在第二个 CSS 块中定义 */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div[data-baseweb="input"] label,
    [data-testid="stSidebar"] label[for*="number"],
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stNumberInput > label,
    [data-testid="stSidebar"] div[data-baseweb="input"] > label,
    [data-testid="stSidebar"] div[data-baseweb="input"] p,
    [data-testid="stSidebar"] p:not([class*="metric"]):not([class*="title"]),
    [data-testid="stSidebar"] [data-baseweb="input"] + *,
    [data-testid="stSidebar"] [class*="stNumberInput"] label,
    [data-testid="stSidebar"] [class*="stNumberInput"] p,
    [data-testid="stSidebar"] div[data-baseweb="input"] ~ * {
        font-weight: 600 !important;
        color: #34495e !important;
    }
    /* 输入框容器和标签字体大小由 CSS 变量控制 */
    [data-testid="stSidebar"] [data-baseweb="input"] { }
    [data-testid="stSidebar"] [data-baseweb="input"] *:not(input):not(button) { font-weight: 600 !important; }
    [data-testid="stSidebar"] label { margin-bottom: 0.2rem !important; }

    /* 侧边栏五个特征输入框数值 */
    [data-testid="stSidebar"] input[type="number"] {
        font-size: 1.1em !important;
        font-weight: bold !important;
        padding: 0.4em 0.5em !important;
        border-radius: 6px !important;
        border: 2px solid #dee2e6 !important;
        transition: border-color 0.3s ease !important;
        margin-bottom: 0.3rem !important;
    }
    [data-testid="stSidebar"] input[type="number"]:focus { border-color: #667eea !important; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important; }
    [data-testid="stSidebar"] div[data-baseweb="input"] { margin-bottom: 0.25rem !important; }

    /* Slider - 基础样式，布局和字体大小在第二个 CSS 块中定义 */
    [data-testid="stSidebar"] .stSlider > div > div { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); }
    /* 滑块标签字体大小由 CSS 变量控制，在第二个 CSS 块中定义 */
    [data-testid="stSidebar"] .stSlider label[data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] .stSlider [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] .stSlider > label:first-child p,
    [data-testid="stSidebar"] .stSlider > div:first-child label p { font-weight: 600 !important; }

    /* 按钮 */
    [data-testid="stSidebar"] button { font-weight: 600 !important; padding: 0.4em 1.5em !important; border-radius: 8px !important; border: none !important; width: 100%; margin-top: 0.3rem !important; transition: transform 0.2s ease, box-shadow 0.2s ease !important; }
    /* 按钮字体大小由 CSS 变量控制，在第二个 CSS 块中定义 */
    [data-testid="stSidebar"] button:last-child,
    [data-testid="stSidebar"] button[data-testid*="baseButton"],
    [data-testid="stSidebar"] > div:last-child button {
        font-weight: 600 !important;
        padding: 0.4em 1.5em !important;
        border-radius: 8px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        width: 100%;
        margin-top: 0.3rem !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    [data-testid="stSidebar"] button:last-child:hover,
    [data-testid="stSidebar"] button[data-testid*="baseButton"]:hover,
    [data-testid="stSidebar"] > div:last-child button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important; }

    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] { margin: 0.2rem 0 !important; }
    [data-testid="stSidebar"] hr { margin: 0.5rem 0 !important; }

    .main-content { padding: 1rem; }

    /* 章节标题 h3 */
    h3 { font-size: 2.5em !important; font-weight: 700 !important; color: #2c3e50 !important; margin-top: 1.5rem !important; margin-bottom: 1rem !important; padding-bottom: 0.5rem; border-bottom: 3px solid #667eea; }

    /* 特征徽章 */
    .feature-badge { display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 8px 16px; border-radius: 20px; font-size: 1.4em; font-weight: 600; margin: 4px; box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3); }

    /* 表格（st.table）基础样式——实际字号稍后会被覆盖为 UI_CSS_FONT_SIZES['table_font'] */
    div[data-testid="stTable"] { margin-bottom: 5rem !important; }
    div[data-testid="stTable"] table { 
        font-size: 16px !important; 
        font-weight: 600 !important;
        border-collapse: collapse !important;
        border: none !important;  /* 去掉表格外边框 */
    }
    div[data-testid="stTable"] table * { font-size: 16px !important; font-weight: 600 !important; }
    div[data-testid="stTable"] thead th { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
        color: white !important; 
        font-size: 16px !important; 
        font-weight: 700 !important; 
        padding: 12px 8px !important;
        border: none !important;  /* 去掉表头边框 */
    }
    div[data-testid="stTable"] tbody td, div[data-testid="stTable"] tbody th { 
        font-size: 16px !important; 
        font-weight: 600 !important; 
        padding: 10px 8px !important;
        border: none !important;  /* 去掉单元格边框 */
    }
    div[data-testid="stTable"] tbody td:first-child, div[data-testid="stTable"] tbody th:first-child { 
        font-size: 16px !important; 
        font-weight: 600 !important;
        border: none !important;  /* 去掉第一列边框 */
    }
    div[data-testid="stTable"] tbody tr:nth-child(even) { background-color: #f8f9fa !important; }
    div[data-testid="stTable"] tbody tr:hover { background-color: #e9ecef !important; transition: background-color 0.2s ease; }

    .metric-card { background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 0.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 追加：仅覆盖字体大小的 CSS 变量与选择器（可在 Python 顶部变量里调节）
st.markdown(
    f"""
    <style>
      :root {{
        --main-title-font-size: {UI_CSS_FONT_SIZES['main_title']};
        --result-banner-title-font-size: {UI_CSS_FONT_SIZES['result_banner_title']};
        --result-banner-detail-font-size: {UI_CSS_FONT_SIZES['result_banner_detail']};
        --sidebar-h2-title-font-size: {UI_CSS_FONT_SIZES['sidebar_h2_title']};
        --sidebar-feature-label-font-size: {UI_CSS_FONT_SIZES['sidebar_feature_label']};
        --sidebar-input-container-font-size: {UI_CSS_FONT_SIZES['sidebar_input_container']};
        --sidebar-input-label-font-size: {UI_CSS_FONT_SIZES['sidebar_input_label']};
        --sidebar-number-input-font-size: {UI_CSS_FONT_SIZES['sidebar_number_input_value']};
        --slider-label-font-size: {UI_CSS_FONT_SIZES['slider_label']};
        --slider-current-value-font-size: {UI_CSS_FONT_SIZES['slider_current_value']};
        --slider-range-font-size: {UI_CSS_FONT_SIZES['slider_range']};
        --slider-range-inner-font-size: {UI_CSS_FONT_SIZES['slider_range_inner']};
        --predict-button-font-size: {UI_CSS_FONT_SIZES['predict_button_text']};
        --section-title-h3-font-size: {UI_CSS_FONT_SIZES['section_title_h3']};
        --feature-badge-font-size: {UI_CSS_FONT_SIZES['feature_badge']};
        --table-font-size: {UI_CSS_FONT_SIZES['table_font']};
        --table-bottom-margin: {UI_CSS_FONT_SIZES['table_bottom_margin']};
      }}
      .main-title {{ font-size: var(--main-title-font-size) !important; }}
      .result-banner {{ font-size: var(--result-banner-title-font-size) !important; }}
      .result-banner > div {{ font-size: var(--result-banner-detail-font-size) !important; }}
      [data-testid="stSidebar"] h2 {{ font-size: var(--sidebar-h2-title-font-size) !important; }}
      /* 五个特征标签 - 使用多个选择器确保覆盖 */
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] label [data-testid="stMarkdownContainer"],
      [data-testid="stSidebar"] label [data-testid="stMarkdownContainer"] p,
      [data-testid="stSidebar"] div[data-baseweb="input"] label,
      [data-testid="stSidebar"] div[data-baseweb="input"] label [data-testid="stMarkdownContainer"] p,
      [data-testid="stSidebar"] label[for*="number"],
      [data-testid="stSidebar"] .stNumberInput label,
      [data-testid="stSidebar"] .stNumberInput label [data-testid="stMarkdownContainer"] p,
      [data-testid="stSidebar"] .stNumberInput > label,
      [data-testid="stSidebar"] .stNumberInput > label [data-testid="stMarkdownContainer"] p,
      [data-testid="stSidebar"] div[data-baseweb="input"] > label,
      [data-testid="stSidebar"] div[data-baseweb="input"] > label [data-testid="stMarkdownContainer"] p,
      [data-testid="stSidebar"] div[data-baseweb="input"] p,
      [data-testid="stSidebar"] p:not([class*="metric"]):not([class*="title"]),
      [data-testid="stSidebar"] [data-baseweb="input"] + *,
      [data-testid="stSidebar"] [class*="stNumberInput"] label,
      [data-testid="stSidebar"] [class*="stNumberInput"] label [data-testid="stMarkdownContainer"] p,
      [data-testid="stSidebar"] [class*="stNumberInput"] p,
      [data-testid="stSidebar"] div[data-baseweb="input"] ~ * {{ font-size: var(--sidebar-feature-label-font-size) !important; }}
      [data-testid="stSidebar"] [data-baseweb="input"] {{ font-size: var(--sidebar-input-container-font-size) !important; }}
      [data-testid="stSidebar"] [data-baseweb="input"] *:not(input):not(button) {{ font-size: var(--sidebar-input-label-font-size) !important; }}
      [data-testid="stSidebar"] input[type="number"] {{ font-size: var(--sidebar-number-input-font-size) !important; font-weight: bold !important; }}
      /* Threshold 滑块标签 - 使用多个选择器确保覆盖 */
      [data-testid="stSidebar"] .stSlider label[data-testid="stWidgetLabel"] p,
      [data-testid="stSidebar"] .stSlider [data-testid="stWidgetLabel"] p,
      [data-testid="stSidebar"] .stSlider [data-testid="stWidgetLabel"] [data-testid="stMarkdownContainer"] p,
      [data-testid="stSidebar"] .stSlider > label:first-child p,
      [data-testid="stSidebar"] .stSlider > label:first-child [data-testid="stMarkdownContainer"] p,
      [data-testid="stSidebar"] .stSlider > div:first-child label p,
      [data-testid="stSidebar"] .stSlider > div:first-child label [data-testid="stMarkdownContainer"] p,
      [data-testid="stSidebar"] .stSlider label p,
      [data-testid="stSidebar"] .stSlider label [data-testid="stMarkdownContainer"] p {{ font-size: var(--slider-label-font-size) !important; font-weight: 600 !important; }}
      
      /* 滑块当前值 */
      [data-testid="stSidebar"] .stSlider > div:nth-child(2),
      [data-testid="stSidebar"] .stSlider > div:nth-child(2) *,
      [data-testid="stSidebar"] .stSlider > div:nth-child(2) [data-testid="stMarkdownContainer"],
      [data-testid="stSidebar"] .stSlider > div:nth-child(2) span,
      [data-testid="stSidebar"] .stSlider > div:nth-child(2) strong,
      [data-testid="stSidebar"] .stSlider > div:not(:first-child):not(:last-child),
      [data-testid="stSidebar"] .stSlider > div:not(:first-child):not(:last-child) * {{ font-size: var(--slider-current-value-font-size) !important; font-weight: 500 !important; }}
      
      /* 滑块范围 */
      [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
      [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"],
      [data-testid="stSidebar"] .stSlider > div:last-child:not([data-testid*="MarkdownContainer"]) {{ font-size: var(--slider-range-font-size) !important; }}
      [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"] *,
      [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] * {{ font-size: var(--slider-range-inner-font-size) !important; }}
      
      /* Predict 按钮 - 使用多个选择器确保覆盖 */
      [data-testid="stSidebar"] button:last-child,
      [data-testid="stSidebar"] button[data-testid*="baseButton"],
      [data-testid="stSidebar"] button[data-testid*="baseButton-secondary"],
      [data-testid="stSidebar"] > div:last-child button,
      [data-testid="stSidebar"] button:last-child [data-testid="stMarkdownContainer"],
      [data-testid="stSidebar"] button:last-child [data-testid="stMarkdownContainer"] p,
      [data-testid="stSidebar"] button[data-testid*="baseButton"] [data-testid="stMarkdownContainer"],
      [data-testid="stSidebar"] button[data-testid*="baseButton"] [data-testid="stMarkdownContainer"] p {{ font-size: var(--predict-button-font-size) !important; }}
      h3 {{ font-size: var(--section-title-h3-font-size) !important; }}
      .feature-badge {{ font-size: var(--feature-badge-font-size) !important; }}
      div[data-testid="stTable"] {{ margin-bottom: var(--table-bottom-margin) !important; }}
      div[data-testid="stTable"] table, div[data-testid="stTable"] table * {{ font-size: var(--table-font-size) !important; }}
      div[data-testid="stTable"] thead th {{ font-size: var(--table-font-size) !important; }}
      div[data-testid="stTable"] tbody td, div[data-testid="stTable"] tbody th {{ font-size: var(--table-font-size) !important; }}
      div[data-testid="stTable"] tbody td:first-child, div[data-testid="stTable"] tbody th:first-child {{ font-size: var(--table-font-size) !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# 行内标签样式与隐藏数字步进按钮（±）
st.markdown(
    f"""
    <style>
      /* 行内特征标签（与输入框同一行显示） */
      .feature-inline-label {{
        font-size: {UI_CSS_FONT_SIZES['feature_inline_label']} !important;
        font-weight: 600 !important;
        color: #34495e !important;
        display: flex; align-items: center; min-height: 42px; line-height: 1.2; /* 与输入框高度对齐，随字号增大自动增高 */
        white-space: nowrap; /* 避免换行 */
      }}

      /* 隐藏浏览器自带的数字输入上下箭头（Chrome/Edge） */
      [data-testid="stSidebar"] input[type=number]::-webkit-outer-spin-button,
      [data-testid="stSidebar"] input[type=number]::-webkit-inner-spin-button {{
        -webkit-appearance: none !important;
        margin: 0 !important;
      }}
      /* 隐藏 Firefox 数字输入上下箭头 */
      [data-testid="stSidebar"] input[type=number] {{
        -moz-appearance: textfield !important;
      }}
      /* 隐藏 BaseWeb/Streamlit 包裹的数字输入上的增减按钮（如果存在） */
      [data-testid="stSidebar"] .stNumberInput button,
      [data-testid="stSidebar"] div[data-baseweb="input"] button {{
        display: none !important;
      }}

      /* ============================================
         滑块字体大小和间距优化：增大字体，增加间距避免遮挡
         ============================================ */
      
      /* 滑块容器：增加垂直间距，保持原有布局 */
      [data-testid="stSidebar"] .stSlider {{
        padding: 0.6em 0 !important;
        margin-bottom: 0.6rem !important;
      }}
      
      /* 当前值：增大字体，增加上下间距，保持原有位置 */
      [data-testid="stSidebar"] .stSlider > div:nth-child(2) {{
        margin: 0.4rem 0 !important;
        padding: 0.2rem 0 !important;
      }}
      
      [data-testid="stSidebar"] .stSlider > div:nth-child(2) [data-testid="stMarkdownContainer"],
      [data-testid="stSidebar"] .stSlider > div:nth-child(2) [data-testid="stMarkdownContainer"] *,
      [data-testid="stSidebar"] .stSlider > div:nth-child(2) span,
      [data-testid="stSidebar"] .stSlider > div:nth-child(2) strong,
      [data-testid="stSidebar"] .stSlider > div:nth-child(2) p {{
        font-size: var(--slider-current-value-font-size) !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
        line-height: 1.5 !important;
      }}
      
      /* 滑块轨道：增加上下间距 */
      [data-testid="stSidebar"] .stSlider > div:nth-child(3) {{
        margin: 0.4rem 0 !important;
      }}
      
      /* 范围值容器：增加顶部间距，确保在滑块下方 */
      [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] {{
        margin-top: 0.5rem !important;
        padding-top: 0.3rem !important;
      }}
      
      [data-testid="stSidebar"] .stSlider > div:last-child:not([data-testid*="MarkdownContainer"]) {{
        margin-top: 0.5rem !important;
        padding-top: 0.3rem !important;
      }}
      
      /* 范围值（0.00 和 1.00）：增大字体 */
      [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
      [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {{
        font-size: var(--slider-range-font-size) !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
      }}
      
      [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"] *,
      [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] * {{
        font-size: var(--slider-range-inner-font-size) !important;
        font-weight: 700 !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# 使用缓存避免重复训练和数据加载
@st.cache_resource
def load_and_compute():
    df = load_data()
    importances = compute_importances(df)
    summary_df = build_summary_df(df, importances)
    top5 = list(importances.sort_values(ascending=False).head(5).index)
    return df, importances, summary_df, top5


df, importances, summary_df, top5 = load_and_compute()

# Main title
st.markdown(
    """
    <div class="main-title">
        🔬 Black Phosphorus Gas Response Predictor
    </div>
    """,
    unsafe_allow_html=True
)

# Feature badges
feature_badges = ''.join([f'<span class="feature-badge">{feat}</span>' for feat in top5])
TOP5_FEATURES_LABEL_FONT_SIZE = UI_CSS_FONT_SIZES['top5_label']
st.markdown(
    f"""
    <div style="margin: 1rem 0 2rem 0;">
        <strong style="font-size: {TOP5_FEATURES_LABEL_FONT_SIZE}; color: #2c3e50;">Top-5 Features:</strong><br>
        {feature_badges}
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 0.1rem 0; margin-top: 0;">
        <h2 style="margin: 0; padding: 0;">⚙️ Input Parameters</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# 气体名称输入（优先）
gas_name = st.sidebar.text_input(
    "🔬 Gas Name",
    value="",
    help="输入气体名称（如 NO2, CO, NH3 等），如果输入了气体名称，将直接从嵌入的数据中查找对应的Response值，而不使用特征判断"
)

gas_response = None
if gas_name and gas_name.strip():
    gas_name_upper = gas_name.strip().upper()
    found_gas = None
    for gas_key, response_val in GAS_RESPONSE_DATA.items():
        if gas_key.upper() == gas_name_upper:
            found_gas = gas_key
            gas_response = response_val
            break
    if gas_response is not None:
        st.sidebar.success(f"✅ 找到气体: {found_gas}, Response = {gas_response}")
    else:
        gas_column = None
        possible_names = ['Gas type', 'Gas Type', 'Gas', 'gas type', 'gas', 'GasName', 'gas_name']
        for col_name in possible_names:
            if col_name in df.columns:
                gas_column = col_name
                break
        if gas_column is None:
            for col in df.columns:
                if 'gas' in str(col).lower():
                    gas_column = col
                    break
        if gas_column:
            try:
                gas_matches = df[df[gas_column].astype(str).str.strip().str.upper() == gas_name_upper]
                if len(gas_matches) > 0:
                    gas_response = int(gas_matches.iloc[0][TARGET])
                    st.sidebar.success(f"✅ 找到气体: {gas_matches.iloc[0][gas_column]}, Response = {gas_response}")
                else:
                    st.sidebar.warning(f"⚠️ 未找到气体名称: {gas_name}")
            except Exception as e:
                st.sidebar.error(f"❌ 查找气体时出错: {str(e)}")
        else:
            st.sidebar.warning(f"⚠️ 未找到气体名称: {gas_name}")

new_sample = {}
if not gas_name or not gas_name.strip():
    for feat in top5:
        col = df[feat]
        vmin, vmax = float(col.min()), float(col.max())
        default = float(col.mean())
        step = 0.0001
        # 行内布局：左侧为特征名，右侧为输入框
        c1, c2 = st.sidebar.columns([1, 2], gap="small")
        with c1:
            # 使用内联 style 直接应用配置的字号，确保不受其他 CSS 干扰
            st.markdown(
                f"<div class='feature-inline-label' style='font-size:{UI_CSS_FONT_SIZES['feature_inline_label']} !important; font-weight:600; color:#34495e'>{feat}</div>",
                unsafe_allow_html=True
            )
        with c2:
            new_sample[feat] = st.number_input(
                label=f"{feat}",
                min_value=vmin,
                max_value=vmax,
                value=default,
                step=step,
                format="%.4f",
                label_visibility="collapsed",
                key=f"num_{feat}"
            )
else:
    for feat in top5:
        new_sample[feat] = 0.0

# 分布与推荐阈值
y = df[TARGET]

def compute_score_distribution(df_data, summary_df_subset):
    scores = []
    for idx, row in df_data.iterrows():
        sample = {feat: row[feat] for feat in summary_df_subset['Feature'].tolist()}
        _, total_score, _ = predict_response_weighted(sample, summary_df_subset, threshold=0.0)
        scores.append(total_score)
    return np.array(scores)

sub_summary_all = summary_df[summary_df['Feature'].isin(top5)].reset_index(drop=True)
all_scores = compute_score_distribution(df, sub_summary_all)
response_scores = compute_score_distribution(df[df[TARGET] == 1], sub_summary_all)
nonresponse_scores = compute_score_distribution(df[df[TARGET] == 0], sub_summary_all)

median_response = np.median(response_scores) if len(response_scores) > 0 else 0.5
median_nonresponse = np.median(nonresponse_scores) if len(nonresponse_scores) > 0 else 0.5
recommended_threshold = (median_response + median_nonresponse) / 2

thr = st.sidebar.slider(
    'Threshold', 
    0.0, 1.0, 
    float(recommended_threshold), 
    0.01,
    help="Adjust the threshold for classification decision"
)

if st.sidebar.button('Predict'):
    if gas_name and gas_name.strip() and gas_response is not None:
        pred = 'Response' if gas_response == 1 else 'Non-Response'
        total_weighted_score = gas_response
        details_df = pd.DataFrame({
            'Feature': ['Gas Lookup'],
            'Value': [gas_name],
            'Weight': [1.0],
            'Score (0-1)': [float(gas_response)],
            'Weighted Score': [float(gas_response)]
        })
        banner_class = 'result-green' if pred == 'Response' else 'result-gray'
        icon = '✅' if pred == 'Response' else '❌'
        st.markdown(
            f"""
            <div class='result-banner {banner_class}'>
                {icon} <strong>Prediction: {pred}</strong><br>
                <div style="font-size: {UI_CSS_FONT_SIZES['result_banner_detail']}; margin-top: 8px; opacity: 0.95;">
                    来源: Excel直接查找 (Gas: <strong>{gas_name}</strong>) | 
                    Response值: <strong>{gas_response}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        gas_info_found = False
        if 'Gas type' in df.columns or any('gas' in str(col).lower() for col in df.columns):
            try:
                gas_column = None
                for col in df.columns:
                    if 'gas' in str(col).lower():
                        gas_column = col
                        break
                if gas_column:
                    gas_matches = df[df[gas_column].astype(str).str.strip().str.upper() == gas_name.strip().upper()]
                    if len(gas_matches) > 0:
                        gas_info = gas_matches.iloc[0]
                        st.markdown('### 📋 Gas Information')
                        info_dict = {
                            'Gas Type': gas_info.get(gas_column, gas_name),
                            'Response': gas_info.get('Response', gas_response),
                        }
                        if 'Site' in gas_info:
                            info_dict['Site'] = gas_info.get('Site', 'N/A')
                        for feat in top5:
                            if feat in gas_info:
                                info_dict[feat] = f"{gas_info[feat]:.4f}"
                        info_df = pd.DataFrame([info_dict])
                        st.table(info_df)
                        gas_info_found = True
            except Exception:
                pass
        if not gas_info_found:
            st.markdown('### 📋 Gas Information')
            info_dict = { 'Gas Type': gas_name, 'Response': gas_response }
            info_df = pd.DataFrame([info_dict])
            st.table(info_df)
    else:
        sub_summary = summary_df[summary_df['Feature'].isin(top5)].reset_index(drop=True)
        pred, total_weighted_score, details_df = predict_response_weighted(new_sample, sub_summary, thr)
        banner_class = 'result-green' if pred == 'Response' else 'result-gray'
        icon = '✅' if pred == 'Response' else '❌'
        st.markdown(
            f"""
            <div class='result-banner {banner_class}'>
                {icon} <strong>Prediction: {pred}</strong><br>
                <div style="font-size: {UI_CSS_FONT_SIZES['result_banner_detail']}; margin-top: 8px; opacity: 0.95;">
                    Weighted Score: <strong>{total_weighted_score:.4f}</strong> | 
                    Threshold: <strong>{thr:.2f}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('### 📋 Feature Decision Details')
        display_df = details_df.copy()
        display_df['Importance'] = display_df['Importance'].apply(lambda x: f'{x:.4f}')
        display_df['Score'] = display_df['Score'].apply(lambda x: f'{x:.4f}')
        display_df['WeightedScore'] = display_df['WeightedScore'].apply(lambda x: f'{x:.6f}')
        display_df['Distance_to_Response'] = display_df['Distance_to_Response'].apply(lambda x: f'{x:.4f}')
        display_df['Distance_to_NonResponse'] = display_df['Distance_to_NonResponse'].apply(lambda x: f'{x:.4f}')
        display_df.columns = ['Feature', 'Value', 'Weight', 'Response Center', 'NonResponse Center', 
                              'Dist to Response', 'Dist to NonResponse', 'Score (0-1)', 'Weighted Score']
        st.table(display_df[['Feature', 'Value', 'Weight', 'Score (0-1)', 'Weighted Score']])
        st.markdown(
            f"""
            <style>
            div[data-testid="stTable"] {{
                margin-bottom: {UI_CSS_FONT_SIZES['table_bottom_margin']} !important;
            }}
            div[data-testid="stTable"] table {{
                font-size: {UI_CSS_FONT_SIZES['table_font']} !important;
                border-collapse: collapse !important;
                border: none !important;  /* 去掉表格外边框 */
            }}
            div[data-testid="stTable"] thead th {{
                font-size: {UI_CSS_FONT_SIZES['table_font']} !important;
                font-weight: 700 !important;
                border: none !important;  /* 去掉表头边框 */
            }}
            div[data-testid="stTable"] tbody td,
            div[data-testid="stTable"] tbody th {{
                font-size: {UI_CSS_FONT_SIZES['table_font']} !important;
                font-weight: 600 !important;
                border: none !important;  /* 去掉单元格边框 */
            }}
            div[data-testid="stTable"] tbody td:first-child,
            div[data-testid="stTable"] tbody th:first-child {{
                font-size: {UI_CSS_FONT_SIZES['table_font']} !important;
                font-weight: 600 !important;
                border: none !important;  /* 去掉第一列边框 */
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown('### 📊 Weighted Score Sum Calculation Visualization')
        fig = plot_weighted_score_waterfall(details_df, total_weighted_score, thr)
        st.pyplot(fig)
        st.markdown('### 📈 Feature Value Ranges Visualization')
        fig2 = plot_ranges(sub_summary, new_sample)
        st.pyplot(fig2)

