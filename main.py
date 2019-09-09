import numpy as np
import sys
from FCM.fuzzy_clustering import FCM
from FCM.visualization import draw_model_2d

sys.path.append('..')


def main():
    f = []
    import random
    # data generation
    for i in range(1000):
        f.append([random.randint(1, 100), random.randint(1, 100)])
    X = np.array(f)
    fcm = FCM(n_clusters=5, m=2, max_iter=15)
    predicted_membership = fcm.learn(X)
    print("u:\n")
    print(predicted_membership)
    print("\n\n")
    draw_model_2d(fcm, data=X, membership=predicted_membership)


if __name__ == '__main__':
    main()
