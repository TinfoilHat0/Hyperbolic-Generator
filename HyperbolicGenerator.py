#mpiexec  -n 3 python3  HyperbolicGenerator.py 1000
import sys
from mpi4py import MPI
from math import *
import numpy as np
import bisect as bs
from operator import itemgetter
from _NetworKit import Graph

class HyperbolicGenerator:
    def __init__(self, n = 1000, k = 6, plexp = 3):
        """ n = # of points, k = avgDegree, plexp = power-law expnonent """
        self.n = n
        self.k  = k
        self.alpha = (plexp-1)/2
        self.R = self.hyperbolicAreaToRadius(n)
        self.thresholdDistance = self.getTargetRadius(n, n*k/2, self.alpha)
        self.stretch = self.thresholdDistance / self.R
        self.factor = 1
        self.bandRatioConstant = 1/2


    def generateFastMPI(self):
        """ Generates a RHG from the supplied args with MPI """
        #0. Distribute the points to processes
        nLoc = int(self.n/size)
        if (self.n % size) > rank:
            nLoc += 1
        #1. Compute band parameters in root process and broadcast to other processes
        c, bands = None, []
        if rank == 0:
            c = self.getBandRadius(self.n, self.thresholdDistance, self.bandRatioConstant)
            for i in range(0, len(c)-1):
                bands.append(Band(c[i], c[i+1]))
        c = comm.bcast(c, root=0)
        bands = comm.bcast(bands, root=0)
        #2. Generate points in disk and insert them to bands locally
        points = self.generatePoints(nLoc, self.stretch, self.alpha, size, rank)
        points.sort(key=itemgetter(0))
        for point in points:
            for b in bands:
                if (point[1] >= b.cLow and point[1] <= b.cHigh):
                    b.insert(point)
                    break
        #3. Generate edges
        edges = set()
        for v in points:
            for b in bands:
                if b.cHigh > v[1]:
                    minTheta, maxTheta = self.getMinMaxTheta(v, b, self.thresholdDistance)
                    slab = self.getPointsWithinAngles(minTheta, maxTheta, b)
                    for w in slab:
                        if v != w:
                            if self.getHyperbolicDistance(v, w) <= self.thresholdDistance:
                                edges.add((v[2], w[2]))




    def generateMPI(self):
        """ Generates a RHG from the supplied args with MPI """
        #0. Divide the points to processes
        nLoc = int(self.n/size)
        nMax = nLoc + 1
        if (self.n % size) > rank:
            nLoc += 1
        #1. Compute band parameters in root process and broadcast to other processes
        c, bands = None, []
        if rank == 0:
            c = self.getBandRadius(self.n, self.thresholdDistance, self.bandRatioConstant)
            for i in range(0, len(c)-1):
                bands.append(Band(c[i], c[i+1]))
        c = comm.bcast(c, root=0)
        bands = comm.bcast(bands, root=0)
        #2. Generate points in disk and insert them to bands locally
        points = self.generatePoints(nLoc, self.stretch, self.alpha)
        points.sort(key=itemgetter(0))
        for point in points:
            for b in bands:
                if (point[1] >= b.cLow and point[1] <= b.cHigh):
                    b.insert(point)
                    break
        #3. Generate edges
        edges = set()
        comm.barrier()
        for i in range(0, nMax):
            for j in range(0, len(bands)):
                comm.barrier()
                bCombined = comm.allgather(bands[j])
                if i < nLoc:
                    minTheta, maxTheta = self.getMinMaxTheta(points[i], bands[j], self.thresholdDistance)
                    for b in bCombined:
                        if b.cHigh > points[i][1]:
                            slab = self.getPointsWithinAngles(minTheta, maxTheta, b)
                            for w in slab:
                                if points[i] != w:
                                    if self.getHyperbolicDistance(points[i], w) <= self.thresholdDistance:
                                        edges.add((points[i][2], w[2]))
        comm.barrier()
        #5. Gather resulting edges in root process. Remove parallel edges. (There should be a better way to do it)
        edgesCombined = comm.gather(edges, root=0)
        if rank == 0:
            g = Graph(self.n)
            temp = set()
            for edges in edgesCombined:
                for edge in edges:
                    if(edge[1], edge[0]) not in temp:
                        temp.add(edge)
            for edge in temp:
                g.addEdge(edge[0], edge[1])
            print(len(g.edges()))
            return g

    def generate(self):
        """ Generates a RHG from the supplied args """
        #1. Generate points in disk
        points = self.generatePoints(self.n, self.stretch, self.alpha)
        #If we sort points here, we don't need to sort bands afterwards
        points.sort(key=itemgetter(0))
        #2. Calculate band radius' and define bands
        c = self.getBandRadius(self.n, self.thresholdDistance, self.bandRatioConstant)
        bands = []
        for i in range(0, len(c)-1):
            bands.append(Band(c[i], c[i+1]))
        #3.Insert points to the bands
        for point in points:
            for b in bands:
                if (point[1] >= b.cLow and point[1] <= b.cHigh):
                    b.insert(point)
                    break
        #4. Insert edges
        edges = set()
        for v in points:
            for b in bands:
                if b.cHigh > v[1]:
                    minTheta, maxTheta = self.getMinMaxTheta(v, b, self.thresholdDistance)
                    slab = self.getPointsWithinAngles(minTheta, maxTheta, b)
                    for w in slab:
                        if v != w:
                            if self.getHyperbolicDistance(v, w) <= self.thresholdDistance:
                                edges.add((v[2], w[2]))
        #5. Remove parallel edges. (There should be a better way to do it)
        g = Graph(self.n)
        temp = set()
        for edge in edges:
            if(edge[1], edge[0]) not in temp:
                temp.add(edge)
        for edge in temp:
            g.addEdge(edge[0], edge[1])
        return g

    def getTargetRadius(self, n, m, alpha = 1.0, epsilon = 0.01):
        """ Calculates edge threshold val. """
        plexp = 2*alpha+1;
        targetAvgDegree = (m/n)*2;
        xiInv = ((plexp-2)/(plexp-1));
        v = targetAvgDegree * (pi/2)*xiInv*xiInv;
        result = 2*log(n / v);
        expected = self.getExpectedDegree(n, alpha, result);
        result = self.searchTargetRadiusForColdGraphs(n, targetAvgDegree, alpha, epsilon);
        return result;

    def getExpectedDegree(self, n, alpha, R):
        gamma = 2*alpha+1
        xi = (gamma-1)/(gamma-2)
        firstSumTerm = exp(-R/2)
        secondSumTerm = exp(-alpha*R)*(alpha*(R/2)*((pi/4)*pow((1/alpha),2)-(pi-1)*(1/alpha)+(pi-2))-1)
        expectedDegree = (2/pi)*xi*xi*n*(firstSumTerm + secondSumTerm)
        return expectedDegree

    def searchTargetRadiusForColdGraphs(self, n, k, alpha, epsilon):
        gamma = 2*alpha+1
        xiInv = ((gamma-2)/(gamma-1))
        v = k * (pi/2)*xiInv*xiInv
        currentR = 2*log(n / v)
        lowerBound = currentR/2
        upperBound = currentR*2
        assert(self.getExpectedDegree(n, alpha, lowerBound) > k)
        assert(self.getExpectedDegree(n, alpha, upperBound) < k)
        while True:
            currentR = (lowerBound + upperBound)/2
            currentK = self.getExpectedDegree(n, alpha, currentR)
            if currentK < k:
                upperBound = currentR
            else:
                lowerBound = currentR
            if abs(self.getExpectedDegree(n, alpha, currentR) - k) <= epsilon:
                break
        return currentR

    def getBandRadius(self, n, thresholdDistance, r):
        """
            We asumme band differences form a geometric series.
            Thus, there is a constant ratio(r) between band length differences
            i.e c2-c1/c1-c0 = c3-c2/c2-c1 = r
        """
        c = [0] #c_0 = 0
        a = thresholdDistance*(1-r)/(1-r**log(n))
        for i in range(1, ceil(log(n))):
            c_i = a*(1-r**i)/(1-r)
            c.append(c_i)
        c.append(thresholdDistance) #c_max = R
        return c

    def generatePoints(self, n, stretch, alpha, pSize = 1, pRank = 0):
        """ Generates n random points in hyperbolic disk """
        points = []
        maxR, minR = self.thresholdDistance, 0
        maxPhi, minPhi = ((pRank+1)*2*pi*)/pSize, (pRank*2*pi)/pSize
        maxcdf, mincdf = cosh(alpha*maxR), cosh(alpha*minR)
        for i in range(0, n):
            angle = np.random.uniform(minPhi, maxPhi)
            random = np.random.uniform(mincdf, maxcdf)
            radius = (acosh(random)/alpha)
            assert(angle <= maxPhi)
            assert(angle >= minPhi)
            assert(radius <= maxR)
            assert(radius >= minR)
            points.append((angle, radius, (i + pRank*n)))
        return points

    def hyperbolicAreaToRadius(self, area):
        return acosh(area/(2*pi)+1)

    def getMinMaxTheta(self, point, band, thresholdDistance):
        """
            Calculates the angles that are enclosing the intersection of the
            hyperbolic disk that is around point v and the bands.
            Calculation is as follows:
            1. For the most inner band, return [0, 2pi]
            2. For other bands, consider the point P which lies on the tangent from origin to the disk of point v.
            Its radial coordinates would be(cHigh, point[1]+a). We're looking for the a.
            We know the distance from point v to P is R. Thus, we can solve the hyperbolic distance of (v, P)
            for a. Then, thetaMax is simply point[1] + a and thetaMin is point[1] - a
        """
        #Most innerband is defined by cLow = 0, cHigh = c[1]
        if band.cLow == 0:
            return (0, 2*pi)
        #For other bands, calculate as described above
        a = (cosh(point[1])*cosh(band.cLow) - cosh(thresholdDistance))/(sinh(point[1])*sinh(band.cLow))
        #handle floating point error
        if a < -1:
            a = -1
        elif a > 1:
            a = 1
        a = acos(a)
        minTheta = point[0] - a
        maxTheta = point[0] + a
        return(minTheta, maxTheta)

    def getPointsWithinAngles(self, minTheta, maxTheta, band):
        """
            Returns the list of points, w, that lies within minTheta and maxTheta
            in the supplied band(That area is called as slab).
        """
        angles = band.angles
        #Case 1: We do not have overlap 2pi, simply put all the points between min and max to the list
        slab = []
        if maxTheta <= 2*pi and minTheta >= 0:
            low = bs.bisect_left(angles, minTheta)
            high = bs.bisect_right(angles, maxTheta)
            slab = band.points[low:high]
        #Case 2: We have 'forward' overlap at 2pi, that is maxTheta > 2pi
        elif maxTheta > 2*pi:
            #1. Get points from minTheta to 2pi
            low = bs.bisect_left(angles, minTheta)
            high = bs.bisect_right(angles, 2*pi)
            slab = band.points[low:high]
            #2. Get points from 0 to maxTheta%2pi
            low = bs.bisect_left(angles, 0)
            maxTheta = maxTheta % (2*pi)
            high = bs.bisect_right(angles, maxTheta)
            slab += band.points[low:high]
        #Case 3: We have 'backward' overlap at 2pi, that is minTheta < 0
        elif minTheta < 0:
            #1. Get points from 2pi - minTheta to 2pi
            minTheta = (2*pi) - minTheta
            low = bs.bisect_left(angles, minTheta)
            high = bs.bisect_right(angles, 2*pi)
            slab = band.points[low:high]
            #2. Get points from 0 to maxTheta
            low = bs.bisect_left(angles, 0)
            high = bs.bisect_right(angles, maxTheta)
            slab += band.points[low:high]

        return slab

    def getHyperbolicDistance(self, u, v):
        """ Returns the hyperbolic distance approximated between the points u and v.
                (as defined in 2010 paper, eqn:6)
        """
        deltaTheta = pi - abs(pi-abs(u[0] - v[0]))
        return u[1] + v[1] + 2*log(deltaTheta/2)

class Band:
    def __init__(self, cLow, cHigh):
        """ Initializes band params. """
        self.cLow = cLow
        self.cHigh = cHigh
        self.points = []
        self.angles = []
    def insert(self, point):
        """ Inserts the supplied point to the band """
        self.points.append(point)
        self.angles.append(point[0])
    def sortByAngular(self):
        """ Sort the band by angular coords """
        self.points.sort(key=itemgetter(0))

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    rhg = None,
    if rank == 0:
        rhg = HyperbolicGenerator(int(sys.argv[1]))
    rhg = comm.bcast(rhg, root=0)
    rhg.generateMPI()
