#!/bin/bash
# ===========================================
# Run Streamlit App
# ===========================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Streamlit Application${NC}"
echo -e "${GREEN}========================================${NC}"

# Create necessary directories
mkdir -p data mlruns onnx_models

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo -e "${YELLOW}Streamlit not found. Installing...${NC}"
    pip install streamlit
fi

echo ""
echo -e "${GREEN}Streamlit app will be available at:${NC}"
echo -e "${GREEN}  Local:   http://localhost:8501${NC}"
echo -e "${GREEN}  Network: http://$(hostname -I | awk '{print $1}'):8501${NC}"
echo ""

# Run Streamlit
streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false