import matplotlib.pyplot as plt

# -----------------------------
# 数据准备
# -----------------------------
epochs = list(range(1, 11))

# ResNet50 数据
resnet50_train_loss = [0.2523, 0.0571, 0.0317, 0.0234, 0.0261, 0.0208, 0.0166, 0.0152, 0.0094, 0.0180]
resnet50_train_acc  = [0.9401, 0.9840, 0.9911, 0.9935, 0.9918, 0.9936, 0.9954, 0.9960, 0.9975, 0.9949]
resnet50_val_loss   = [0.1075, 0.0859, 0.0950, 0.1068, 0.1220, 0.0968, 0.1242, 0.1156, 0.1471, 0.1336]
resnet50_val_acc    = [0.9720, 0.9766, 0.9753, 0.9710, 0.9710, 0.9766, 0.9682, 0.9725, 0.9664, 0.9712]
resnet50_test_loss  = 0.1119
resnet50_test_acc   = 0.9743

# Simple CNN 数据
simplecnn_train_loss = [1.8386, 1.4628, 1.2367, 1.0717, 0.9205, 0.7760, 0.6398, 0.5370, 0.4491, 0.3681]
simplecnn_train_acc  = [0.3569, 0.4992, 0.5793, 0.6358, 0.6868, 0.7352, 0.7825, 0.8188, 0.8461, 0.8762]
simplecnn_val_loss   = [1.5279, 1.3358, 1.1944, 1.1155, 1.1171, 1.0674, 1.1159, 1.1012, 1.1214, 1.1385]
simplecnn_val_acc    = [0.4880, 0.5497, 0.5958, 0.6172, 0.6190, 0.6465, 0.6414, 0.6490, 0.6605, 0.6653]
simplecnn_test_loss  = 1.1334
simplecnn_test_acc   = 0.6515

# ViT 数据
vit_train_loss = [0.1652, 0.0318, 0.0171, 0.0219, 0.0165, 0.0140, 0.0155, 0.0122, 0.0098, 0.0123]
vit_train_acc  = [0.9567, 0.9912, 0.9955, 0.9940, 0.9949, 0.9957, 0.9957, 0.9963, 0.9972, 0.9962]
vit_val_loss   = [0.0835, 0.0951, 0.1190, 0.1278, 0.1027, 0.1140, 0.1525, 0.1298, 0.2159, 0.1160]
vit_val_acc    = [0.9768, 0.9733, 0.9692, 0.9656, 0.9753, 0.9697, 0.9620, 0.9694, 0.9475, 0.9725]
vit_test_loss  = 0.1384
vit_test_acc   = 0.9687

# -----------------------------
# 绘制训练和验证曲线
# -----------------------------
plt.figure(figsize=(16, 6))

# 子图1：Loss 曲线
plt.subplot(1, 2, 1) #设置为1行2列的第1个子图
# ResNet50
plt.plot(epochs, resnet50_train_loss, 'o-', color='blue', label='ResNet50 Train Loss')
# Simple CNN
plt.plot(epochs, simplecnn_train_loss, 's-', color='green', label='Simple CNN Train Loss')
# ViT
plt.plot(epochs, vit_train_loss, '^-', color='red', label='ViT Train Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss Curve')
plt.legend()
plt.grid(True)

################################################################################
plt.subplot(1, 2, 2)
# ResNet50
plt.plot(epochs, resnet50_val_loss, 'o--', color='blue', label='ResNet50 Val Loss')
# Simple CNN
plt.plot(epochs, simplecnn_val_loss, 's--', color='green', label='Simple CNN Val Loss')
# ViT
plt.plot(epochs, vit_val_loss, '^--', color='red', label='ViT Val Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Val Loss Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

################################################################################

# 子图2：Accuracy 曲线
plt.subplot(1, 2, 1)
# ResNet50
plt.plot(epochs, resnet50_train_acc, 'o-', color='blue', label='ResNet50 Train Acc')
# Simple CNN
plt.plot(epochs, simplecnn_train_acc, 's-', color='green', label='Simple CNN Train Acc')
# ViT
plt.plot(epochs, vit_train_acc, '^-', color='red', label='ViT Train Acc')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.grid(True)

#################################################################################
plt.subplot(1, 2, 2)
# ResNet50
plt.plot(epochs, resnet50_val_acc, 'o--', color='blue', label='ResNet50 Val Acc')
# Simple CNN
plt.plot(epochs, simplecnn_val_acc, 's--', color='green', label='Simple CNN Val Acc')
# ViT
plt.plot(epochs, vit_val_acc, '^--', color='red', label='ViT Val Acc')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# -----------------------------
# 绘制测试结果对比（条形图）
# -----------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
models = ['ResNet50', 'Simple CNN', 'ViT']

# 测试 Loss 对比（设置颜色：蓝、绿、红）
ax[0].bar(models, [resnet50_test_loss, simplecnn_test_loss, vit_test_loss],
          color=['blue', 'green', 'red'])
ax[0].set_title('Test Loss Comparison')
ax[0].set_ylabel('Loss')
ax[0].grid(True, axis='y')

# 测试 Accuracy 对比（设置颜色：蓝、绿、红）
ax[1].bar(models, [resnet50_test_acc, simplecnn_test_acc, vit_test_acc],
          color=['blue', 'green', 'red'])
ax[1].set_title('Test Accuracy Comparison')
ax[1].set_ylabel('Accuracy')
ax[1].grid(True, axis='y')

plt.tight_layout()
plt.show()
