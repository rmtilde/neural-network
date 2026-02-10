
def plot_decision_boundary(X, y,model, classes, h=0.01):
    import numpy as np
    import matplotlib.pyplot as plt

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    output = model.forward(grid)


    Z = np.argmax(output, axis=1).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.rainbow)

    # Z = np.max(probs, axis=1)
    # Z_class = np.argmax(probs, axis=1) 
    # Z = Z.reshape(xx.shape)
    # Z_class = Z_class.reshape(xx.shape)
    # plt.figure()
    # plt.contourf(xx, yy, Z_class, alpha=0.3, cmap=plt.cm.rainbow)
    # plt.imshow(Z, extent=(x_min, x_max, y_min, y_max),
    #            origin='lower', cmap='Greys', alpha=0.2, aspect='auto')
    

    
    for class_number in range(classes):
        plt.scatter(X[y==class_number, 0], X[y==class_number, 1],
                    label=f'Class {class_number}', edgecolor='k')
        
    plt.show()