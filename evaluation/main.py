# main.py
from evaluation_manager import EvaluationManager
from accuracy_evaluation import evaluate_accuracy
from f1_evaluation import evaluate_f1

def main():
    manager = EvaluationManager()
    manager.register_evaluation("accuracy", evaluate_accuracy)
    manager.register_evaluation("f1_score", evaluate_f1)

    # 模拟预测和标签数据
    predictions = [0, 1, 0, 1]
    labels = [0, 1, 1, 1]

    # 执行单一评估
    accuracy = manager.evaluate("accuracy", predictions, labels)
    print(f"Accuracy: {accuracy}")

    # 执行所有评估
    results = manager.evaluate_all(predictions, labels)
    print("All Evaluation Results:", results)

if __name__ == "__main__":
    main()