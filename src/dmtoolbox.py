import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def k_mean(points_n, clusters_n, iteration_n, csv):
    #points_n = 500
    #clusters_n = 3
    #iteration_n = 100

    #points = tf.constant(np.random.uniform(0, 10, (points_n, 2)))
    points = tf.constant(csv)
    centroids = tf.Variable(tf.slice(tf.random.shuffle(points), [0, 0], [clusters_n, -1]))

    points_expanded = tf.expand_dims(points, 0)
    centroids_expanded = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
    assignments = tf.argmin(distances, 0)
    update_centroids = centroids


    @tf.function
    def sessrun(update_centroids, centroids, points, assignments):
      points_expanded = tf.expand_dims(points, 0)
      centroids_expanded = tf.expand_dims(centroids, 1)
      distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
      assignments = tf.argmin(distances, 0)
      means = []
      for c in range(clusters_n):
          means.append(tf.reduce_mean(        
            tf.gather(points, 
                      tf.reshape(
                        tf.where(
                          tf.equal(assignments, c)
                        ),[1,-1])
                    ),[1]))
      new_centroids = tf.concat(means, 0)
      update_centroids = centroids.assign(new_centroids)  
      return update_centroids, centroids, points, assignments

    for step in range(iteration_n):
    #  print(centroid_values)
      [_, centroid_values, points_values, assignment_values] = sessrun(update_centroids, centroids, points, assignments)

      
    print("centroids", centroid_values)

    plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)
    plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
    plt.show()