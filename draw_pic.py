import pandas as pd
import matplotlib.pyplot as plt

def draw_accuracy_pic(models, data):
    # 绘制折线图
    fig,ax=plt.subplots(figsize=(12,8), dpi=100)
    epochs = range(1, 101)  # epoch 从1到100

    for i, model in enumerate(models):
        plt.plot(epochs, data[i], label=model)

    plt.title('Accuracy for Two Models', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 创建子图
    ax_inset = ax.inset_axes([0.25, 0.2, 0.4, 0.4])
    for i, model in enumerate(models):
        ax_inset.plot(epochs[90:], data[i][90:], label=model)

    ax_inset.set_xlabel('Epoch')
    ax_inset.set_ylabel('Accuracy (%)')
    ax_inset.set_xlim(90, 101)
    ax_inset.set_ylim(89, 94)
    ax_inset.yaxis.grid(True, linestyle='-.')
    ax.indicate_inset_zoom(ax_inset, edgecolor="black",alpha=0.8)


    # 创建子图
    ax_inset02 = ax.inset_axes([0.75, 0.4, 0.2, 0.2])
    for i, model in enumerate(models):
        ax_inset02.plot(epochs[90:], data[i][90:], label=model)

    ax_inset02.set_xlabel('Epoch')
    ax_inset02.set_ylabel('Accuracy (%)')
    ax_inset02.set_xlim(90, 101)
    ax_inset02.set_ylim(97.1, 97.4)
    ax_inset02.yaxis.grid(True, linestyle='-.')
    ax.indicate_inset_zoom(ax_inset02, edgecolor="red",alpha=0.8)

    plt.show()


if __name__=='__main__':
    # 读取 Excel 文件
    file_path = 'log/new_log_accuracy.xlsx'
    df = pd.read_excel(file_path, header=None)

    # 提取模型名称和数据
    models = []
    data = []

    for i in range(1, len(df)):
        model_name = df.iloc[i, 0].strip()  # 提取模型名称
        model_data = df.iloc[i, 1:].values  # 提取准确率数据
        models.append(model_name)
        data.append(model_data)

    draw_accuracy_pic(models,data)
    # for i in range(len(models)):
    #     print(models[i], max(data[i]))