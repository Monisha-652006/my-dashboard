import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import networkx as nx
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from skimage.metrics import structural_similarity as ssim

st.set_page_config(layout="wide")
st.title("🌍 EcoImpact AI – Smart Environmental Intelligence System")

# -------------------------
# TORCH MODEL
# -------------------------
class LandClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*28*28,128), nn.ReLU(),
            nn.Linear(128,4)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)

model = LandClassifier()
if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

classes = ["Forest","Water","Urban","Open Land"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------------------------
# ML MODEL
# -------------------------
MODEL_PATH = "model.pkl"

def train_model():
    X = np.array([[70,10,20],[60,15,25],[40,25,35],[20,35,45]])
    y = np.array([0,1,1,2])
    model = RandomForestClassifier()
    model.fit(X,y)
    joblib.dump(model,MODEL_PATH)
    return model

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train_model()

# -------------------------
# HELPERS
# -------------------------
def safe_open(file):
    img = Image.open(file)
    return img.convert("RGB")

def classify(img):
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(img)
    return classes[pred.argmax().item()]

# 🔥 LAND ANALYSIS (NEW)
def analyze_land(img):
    img = cv2.resize(img,(300,300))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    forest_mask = cv2.inRange(hsv,(35,40,40),(85,255,255))
    water_mask = cv2.inRange(hsv,(90,50,50),(140,255,255))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    urban_mask = cv2.inRange(gray,180,255)

    total = img.shape[0]*img.shape[1]

    forest = np.sum(forest_mask>0)/total*100
    water = np.sum(water_mask>0)/total*100
    urban = np.sum(urban_mask>0)/total*100

    return forest, water, urban

# 🔥 CHANGE DETECTION
def detect_change(img1,img2):
    img1 = cv2.resize(img1,(300,300))
    img2 = cv2.resize(img2,(300,300))

    g1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    score,diff = ssim(g1,g2,full=True)
    diff = (diff*255).astype("uint8")

    _,thresh = cv2.threshold(diff,30,255,cv2.THRESH_BINARY_INV)

    overlay = img1.copy()
    overlay[thresh==255] = [0,0,255]

    output = cv2.addWeighted(img1,0.7,overlay,0.3,0)

    change_percent = (np.sum(thresh==255)/thresh.size)*100

    return output, thresh, change_percent

def cost_map(image):
    img = np.array(image.resize((100,100)))
    return np.mean(img,axis=2)

def astar_route(cost):
    rows,cols = cost.shape
    G = nx.grid_2d_graph(rows,cols)

    for (x,y) in G.nodes:
        G.nodes[(x,y)]["weight"] = cost[x,y]

    path = nx.astar_path(G,(0,0),(rows-1,cols-1))

    route = np.zeros((rows,cols))
    for p in path:
        route[p]=1
    return route

def generate_solution(forest_loss, urban_growth):
    if forest_loss > 15:
        return "🚨 Severe deforestation → Immediate reforestation required"
    elif urban_growth > 10:
        return "⚠ Urban expansion rising → Control development"
    else:
        return "✅ Ecosystem stable → Maintain conservation"

# -------------------------
# SIDEBAR (UPLOAD ONCE)
# -------------------------
st.sidebar.header("Upload Images")

img1_file = st.sidebar.file_uploader("Upload Image 1")
img2_file = st.sidebar.file_uploader("Upload Image 2")

lat = st.sidebar.number_input("Latitude",value=11.13)
lon = st.sidebar.number_input("Longitude",value=78.66)

img1 = safe_open(img1_file) if img1_file else None
img2 = safe_open(img2_file) if img2_file else None

menu = st.sidebar.radio(
    "Select Module",
    ["Dashboard","Land Classification","Change Detection","Route Optimization","Future Prediction"]
)

# -------------------------
# DASHBOARD
# -------------------------
if menu=="Dashboard":

    if img1:
        st.subheader("🧠 Classification")
        st.image(img1,width=300)
        st.success(classify(img1))

    if img1 and img2:
        st.subheader("🔥 Change Detection")

        img1_np = np.array(img1)
        img2_np = np.array(img2)

        output, thresh, change = detect_change(img2_np,img1_np)

        st.image(output,width=500)
        st.metric("Change %",f"{change:.2f}%")

# -------------------------
# SEPARATE MODULES
# -------------------------
elif menu=="Land Classification" and img1:
    st.image(img1)
    st.success(classify(img1))

elif menu=="Change Detection" and img1 and img2:
    output,_,change = detect_change(np.array(img2),np.array(img1))
    st.image(output)
    st.metric("Change %",f"{change:.2f}%")

elif menu=="Route Optimization" and img1:
    cost = cost_map(img1)
    route = astar_route(cost)
    fig,ax = plt.subplots()
    ax.imshow(cost,cmap="viridis")
    ax.imshow(route,cmap="hot",alpha=0.6)
    st.pyplot(fig)

# -------------------------
# FUTURE PREDICTION (UPDATED)
# -------------------------
elif menu=="Future Prediction" and img1 and img2:

    img1_np = np.array(img1)
    img2_np = np.array(img2)

    f1,w1,u1 = analyze_land(img1_np)
    f2,w2,u2 = analyze_land(img2_np)

    forest_loss = f1 - f2
    water_change = w2 - w1
    urban_growth = u2 - u1

    st.subheader("📊 Environmental Metrics")

    c1,c2,c3 = st.columns(3)
    c1.metric("🌳 Forest Loss %", f"{forest_loss:.2f}%")
    c2.metric("🌊 Water Change %", f"{water_change:.2f}%")
    c3.metric("🏙 Urban Growth %", f"{urban_growth:.2f}%")

    _,_,change = detect_change(img2_np,img1_np)

    st.subheader("📈 Future Prediction")

    model_ml = load_model()
    pred = model_ml.predict([[change, urban_growth, forest_loss]])[0]

    if pred==2:
        st.error("🚨 High Risk Future")
    elif pred==1:
        st.warning("⚠ Moderate Risk")
    else:
        st.success("✅ Stable Future")

    # 🌳 Forest Intelligence
    st.subheader("🌳 Forest Intelligence")

    species = int(300 - forest_loss*2)
    trees_cut = int(forest_loss * 1000)
    habitat_loss = max(0, forest_loss * 1.5)

    c1,c2,c3 = st.columns(3)
    c1.metric("🌱 Species", species)
    c2.metric("🌳 Trees Cut", trees_cut)
    c3.metric("🐾 Habitat Loss %", f"{habitat_loss:.2f}")

    # 💡 AI Suggestion
    st.subheader("💡 AI Suggestion")
    st.info(generate_solution(forest_loss, urban_growth))