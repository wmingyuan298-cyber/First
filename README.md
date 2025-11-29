# BP Response Predictor 使用指南

## 环境配置

1. 确保已安装 Python 3.7 或更高版本
2. 安装所需的依赖包：
   ```bash
   pip install streamlit numpy pandas matplotlib scikit-learn openpyxl
   ```
   （可选）如果需要 SHAP 功能，可以安装：
   ```bash
   pip install shap
   ```

## 运行代码

在终端中运行以下命令启动应用：

```bash
streamlit run app_bp_response_adjustable.py
```

应用会自动在浏览器中打开（通常是 `http://localhost:8501`）

## 使用方法

1. **输入特征值**：在左侧边栏中，输入以下五种特征的值：
   - Eads
   - ℇp
   - VBM
   - CBM
   - Ef

2. **或输入气体名称**：也可以直接在 "Gas Name" 输入框中输入气体名称（如 NO2, CO, NH3 等），系统会直接从数据中查找对应的 Response 值。

3. **调整阈值**：使用滑块调整 Decision Threshold（决策阈值），默认值会根据数据自动计算。

4. **查看预测结果**：点击 "Predict" 按钮后，页面会显示：
   - 预测结果（Response 或 Non-Response）
   - 加权分数和阈值信息
   - 特征决策详情表格
   - 加权分数计算可视化图表
   - 特征值范围可视化图表

## 注意事项

- 确保 `BP-qiti.xlsx` 数据文件与代码文件在同一目录下
- 输入的特征值应在合理范围内，系统会根据数据范围自动限制输入
- 如果输入了气体名称，系统会优先使用气体查找功能，而不是特征值预测

