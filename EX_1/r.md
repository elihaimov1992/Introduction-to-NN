# Mini-Project Number 2.  Kohonen algorithm

## Authors

* **Eli Haimov (ID. 308019306)**
* **Warda Essa (ID. 208642793)**
* **Omar Essa (ID. 315535435)**

## Documentation
Implementation the Kohonen algorithm to perform the following tasks:

For each of these show "snapshots" of the placement of the neurons at different number of iterations of the algorithm.
It may be best to start the initial "locations" of the neurons as small random `(x,y)` values (i.e. all between `-.2 < x,y  < .2`).

* A. Given a sequence of  30 neurons arranged in a straight line;  take data from a circle of radius 2 (centered at the origin), 
with the data points chosen uniformly and randomly within the circle.

* B. Do the same when the 30 neurons have the topology of a circle.

* C. Do the same for 25 neurons arranged in a 5 x 5 topology.

* D. Now repeat A, B, and C  above but where the data is chosen randomly but with a probability proportional to the distance from the center.

* E. Now repeat A and B  but where there are two concentric circles  centered at the origin  one with radious 2 and one with radius 4; 
and all data is chosen from the ring between radius 2 and radius 4.

## Section A:

#### Explanation:
As we can see at first we implanted the neurons in a line shape at random, for each iteration a data point is chosen randomly and 
then with the algorithms we coded each time we calculate the closest neuron to the data point and that will be our winner neuron.

At fist as you can see after two iterations the neurons are still close to each other the winner neuron did affect some of the 
neurons to move with him a lot but there is still some who were affected by him less.

Used 30 neurons, the neurons affecting each other less than when there was less neurons connected to each other.

As we do more iterations 10 ,50 ,100 we can see that the winner neuron does move all neuron line to the random data point we chose 
to each iteration ( the neurons are all over the place.

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)
