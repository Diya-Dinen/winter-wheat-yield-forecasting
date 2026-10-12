import matplotlib.pyplot as plt
import matplotlib.patches as patches

def add_box(ax, text, xy, width, height, facecolor='#ffffff', edgecolor='#000000', boxstyle="round,pad=0.3", fontsize=10, weight='normal'):
    box = patches.FancyBboxPatch(
        (xy[0] - width/2, xy[1] - height/2), 
        width, height, 
        boxstyle=boxstyle,
        edgecolor=edgecolor, facecolor=facecolor, linewidth=1.2
    )
    ax.add_patch(box)
    ax.text(xy[0], xy[1], text, ha='center', va='center', fontsize=fontsize, weight=weight, fontfamily='sans-serif')
    return xy[0], xy[1] - height/2, xy[0], xy[1] + height/2, xy[0] - width/2, xy[0] + width/2

def add_diamond(ax, text, xy, width, height, facecolor='#ffffff', edgecolor='#000000'):
    verts = [
        (xy[0], xy[1] + height/2), # Top
        (xy[0] + width/2, xy[1]),  # Right
        (xy[0], xy[1] - height/2), # Bottom
        (xy[0] - width/2, xy[1])   # Left
    ]
    poly = patches.Polygon(verts, closed=True, edgecolor=edgecolor, facecolor=facecolor, linewidth=1.2)
    ax.add_patch(poly)
    ax.text(xy[0], xy[1], text, ha='center', va='center', fontsize=9, fontfamily='sans-serif')
    return xy[0], xy[1] - height/2, xy[0], xy[1] + height/2, xy[0] - width/2, xy[0] + width/2

def add_arrow(ax, start, end, text=None, text_offset=(0,0)):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(facecolor='black', edgecolor='black', width=1.0, headwidth=6, headlength=8, shrink=0))
    if text:
        ax.text((start[0]+end[0])/2 + text_offset[0], (start[1]+end[1])/2 + text_offset[1], 
                text, ha='center', va='center', fontsize=9, fontfamily='sans-serif', backgroundcolor='white')

# =========================================================================
# 1. Overall System Architecture
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 10))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)

# Colors matching a clean IEEE style
c_bg = '#f9f9f9'
c_edge = '#333333'

box_w = 7
box_h = 0.8

# Data Layer
y = 11
add_box(ax, "HISTORICAL AGRICULTURAL & WEATHER DATA\n(NOAA / NASS APIs)", (5, y), box_w, box_h, c_bg, c_edge, weight='bold')

y -= 1.3
add_box(ax, "PREPROCESSING & FEATURE ENGINEERING\n(40 Agronomic Growth-Stage Variables)", (5, y), box_w, box_h, c_bg, c_edge, weight='bold')
add_arrow(ax, (5, y + 1.3 - 0.4), (5, y + 0.4))

y -= 1.3
add_box(ax, "DRIFT DETECTION ENGINE\n(KS-Test, PSI, Mahalanobis Distance, PCA)", (5, y), box_w, box_h, c_bg, c_edge, weight='bold')
add_arrow(ax, (5, y + 1.3 - 0.4), (5, y + 0.4))

y -= 1.3
add_box(ax, "ADAPTIVE MODEL STRATEGY CONTROLLER\n(Bias Correction / Windowed Retraining)", (5, y), box_w, box_h, c_bg, c_edge, weight='bold')
add_arrow(ax, (5, y + 1.3 - 0.4), (5, y + 0.4))

y -= 1.3
add_box(ax, "MACHINE LEARNING YIELD MODEL\n(XGBoost Regressor)", (5, y), box_w, box_h, c_bg, c_edge, weight='bold')
add_arrow(ax, (5, y + 1.3 - 0.4), (5, y + 0.4))

y -= 1.3
add_box(ax, "UNCERTAINTY ESTIMATION\n(MAPIE Split Conformal Prediction)", (5, y), box_w, box_h, c_bg, c_edge, weight='bold')
add_arrow(ax, (5, y + 1.3 - 0.4), (5, y + 0.4))

y -= 1.3
add_box(ax, "APPLICATION PRESENTATION LAYER\n(FastAPI Backend / Streamlit Dashboard)", (5, y), box_w, box_h, c_bg, c_edge, weight='bold')
add_arrow(ax, (5, y + 1.3 - 0.4), (5, y + 0.4))

plt.tight_layout()
plt.savefig("fig_system_architecture.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# =========================================================================
# 2. Crop Yield Forecasting Workflow
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 12))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)

y = 13
box_w = 4.5
box_h = 0.8

# Start
add_box(ax, "New Environmental &\nWeather Data", (5, y), box_w, box_h, c_bg, c_edge)
y -= 1.3

add_box(ax, "Data Preprocessing &\nFeature Preparation", (5, y), box_w, box_h, c_bg, c_edge)
add_arrow(ax, (5, y + 1.3 - 0.4), (5, y + 0.4))
y -= 1.3

add_box(ax, "Statistical Drift Detection", (5, y), box_w, box_h, c_bg, c_edge)
add_arrow(ax, (5, y + 1.3 - 0.4), (5, y + 0.4))
y -= 1.5

# Diamond
add_diamond(ax, "Significant Drift\nDetected?", (5, y), 4.5, 1.4, c_bg, c_edge)
add_arrow(ax, (5, y + 1.5 - 0.4), (5, y + 0.7))

# Yes Branch
add_box(ax, "Apply Adaptive Strategy\n(Bias Correction / Retrain)", (8, y - 2), 3.5, 0.8, c_bg, c_edge)
# Arrow right from diamond, then down
ax.annotate('', xy=(8, y - 1.6), xytext=(7.25, y), arrowprops=dict(facecolor='black', edgecolor='black', width=1.0, headwidth=6, headlength=8, shrink=0))
ax.text(7.0, y + 0.2, "Yes", ha='center', va='center', fontsize=9, backgroundcolor='white')

# No Branch
add_box(ax, "Standard XGBoost\nYield Prediction", (2, y - 2), 3.5, 0.8, c_bg, c_edge)
ax.annotate('', xy=(2, y - 1.6), xytext=(2.75, y), arrowprops=dict(facecolor='black', edgecolor='black', width=1.0, headwidth=6, headlength=8, shrink=0))
ax.text(3.0, y + 0.2, "No", ha='center', va='center', fontsize=9, backgroundcolor='white')

# Join Back
y -= 3.5
add_box(ax, "MAPIE Conformal Prediction\n(Uncertainty Estimation)", (5, y), 5.5, 0.8, c_bg, c_edge)
ax.annotate('', xy=(4.5, y + 0.4), xytext=(2, y + 1.5 - 0.4), arrowprops=dict(facecolor='black', edgecolor='black', width=1.0, headwidth=6, headlength=8, shrink=0))
ax.annotate('', xy=(5.5, y + 0.4), xytext=(8, y + 1.5 - 0.4), arrowprops=dict(facecolor='black', edgecolor='black', width=1.0, headwidth=6, headlength=8, shrink=0))

y -= 1.3
add_box(ax, "Final Yield Forecast\n+ Prediction Interval", (5, y), 5.5, 0.8, c_bg, c_edge)
add_arrow(ax, (5, y + 1.3 - 0.4), (5, y + 0.4))

y -= 1.3
add_box(ax, "FastAPI & Streamlit\nDashboard Output", (5, y), 5.5, 0.8, c_bg, c_edge)
add_arrow(ax, (5, y + 1.3 - 0.4), (5, y + 0.4))

plt.tight_layout()
plt.savefig("fig_forecasting_workflow.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
