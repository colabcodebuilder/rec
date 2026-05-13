#!/bin/bash
# ===========================================
# Complete Workflow: Train, Evaluate, Serve
# ===========================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Healthcare Recommendation System${NC}"
echo -e "${GREEN}Complete Workflow${NC}"
echo -e "${GREEN}========================================${NC}"

# Step 1: Setup
echo -e "\n${YELLOW}[1/5] Setting up directories...${NC}"
mkdir -p data mlruns

# Step 2: Seed database
echo -e "\n${YELLOW}[2/5] Seeding database...${NC}"
python main.py seed
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to seed database${NC}"
    exit 1
fi

# Step 3: Train model (logs to MLflow)
echo -e "\n${YELLOW}[3/5] Training model and logging to MLflow...${NC}"
python main.py train
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to train model${NC}"
    exit 1
fi

# Step 4: Evaluate
echo -e "\n${YELLOW}[4/5] Running evaluation...${NC}"
python main.py evaluate
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to evaluate${NC}"
    exit 1
fi

# Step 5: Start API server
echo -e "\n${YELLOW}[5/5] Starting API server...${NC}"
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Access URLs:                                      ║${NC}"
echo -e "${CYAN}║  API:        http://localhost:8000                 ║${NC}"
echo -e "${CYAN}║  Docs:       http://localhost:8000/docs            ║${NC}"
echo -e "${CYAN}║  Recommend:  http://localhost:8000/recommend/P-001 ║${NC}"
echo -e "${CYAN}║  MLflow UI:  http://localhost:5000                 ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${RED}To start MLflow UI in another terminal:${NC}"
echo -e "${RED}  bash run_mlflow_local.sh${NC}"
echo ""

python main.py api