from graphics import *
import numpy as np
import pylab as plt
def mazeFind(V):
    #Orientations are 'u','d','l','r'
    def getNextOrientAndPos(currentOrient,currentPos,V):
    #check if you go forward(wall on right continues), turn the corner to the right, or turn left since wall is in front
        if currentOrient=='d':
            if V[currentPos+[0,1]]==1 and V[currentPos+[-1,1]]==0: #checks if path continues down
                return 'd',currentPos+[0,1]#step player one down
            if V[currentPos+[0,1]]==0: #checks if path goes right
                return 'r',currentPos  #turn the player
            if V[currentPos+[0,1]]==1 and V[currentPos+[-1,1]]==1: #checks if path goes left
                return 'l',currentPos+[-1,1]
            if V[currentPos+[0,1]]==0 and V[currentPos+[1,0]]==0: #checks if dead end 
                return 'r',currentPos #turn the player
            
    return 0
    
def union(lowerLabel,higherLabel,classes):
    classes[higherLabel],classes=find(lowerLabel,classes)
    return classes
def find(label,classes):
    z=label
    while classes[label]!=label:
        label=classes[label]
    #for k in np.linspace(z,len(classes)-1,len(classes)-z,dtype=int):
    #    if classes[k]==z:
    #        classes[k]=label
    return label,classes

def findClusters(grid,win):
    classes = np.linspace(0,np.shape(grid)[0]*np.shape(grid)[1]-1,np.shape(grid)[0]*np.shape(grid)[1],dtype=int)
    labels  = np.zeros((np.shape(grid)[0],np.shape(grid)[1]),dtype=int)
    largestClass=0
    for i in range(np.shape(grid)[0]-1):
        i=i+1
        for j in range(np.shape(grid)[1]-1):
            j=j+1
            if grid[i,j]==1:
                left, above = grid[i-1,j], grid[i,j-1]
                leftLabel, aboveLabel = labels[i-1,j], labels[i,j-1]
                if left==0 and above==0:
                    largestClass=largestClass+1
                    labels[i,j]=largestClass
                if left==1 and above==0:
                    labels[i,j],classes=find(leftLabel,classes)
                if left==0 and above==1:
                    labels[i,j],classes=find(aboveLabel,classes)
                if left==1 and above==1:
                    if leftLabel==aboveLabel:
                        labels[i,j]=aboveLabel
                    elif aboveLabel>leftLabel:

                        classes=union(leftLabel,aboveLabel,classes)
                        labels[i,j],classes=find(leftLabel,classes)
                    else:
                        classes=union(aboveLabel,leftLabel,classes)
                        labels[i,j],classes=find(aboveLabel,classes)
    for i in range(np.shape(grid)[0]-1):
        i=i+1
        for j in range(np.shape(grid)[1]-1):
            j=j+1
            labels[i,j]=find(labels[i,j],classes)[0]
    return labels,classes
    
def clusterColor(i):
    if i==0:
        return (255,255,255)
    i=i+1
    return (int(np.mod(40*i,155)),int(np.mod(50*i,155)),int(np.mod(30*i,155)))
def drawRect(v,occupied,L,win,color):
    c = Rectangle(Point(L*v[0],L*v[1]), Point(L*(v[0]+1),L*(v[1]+1)))
    c.setFill(color_rgb(color[0], color[1], color[2]))
    #if not occupied:
    #    c.setOutline("white")
    c.draw(win)
def rotMat(angle):
    return np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
def drawHexagon(v,occupied,L,win,color):
    TL = np.array([-L/2/np.sqrt(3),-L/2])
    TR = rotMat(np.pi/3.0)@ TL
    R = rotMat(np.pi/3.0)@ TR
    BR = rotMat(np.pi/3.0)@ R
    BL = rotMat(np.pi/3.0)@ BR
    L = rotMat(np.pi/3.0)@ BL
    
    c = Polygon(Point(v[0]+TL[0],v[1]+TL[1]), Point(v[0]+TR[0],v[1]+TR[1]),Point(v[0]+R[0],v[1]+R[1]), Point(v[0]+BR[0],v[1]+BR[1]),Point(v[0]+BL[0],v[1]+BL[1]), Point(v[0]+L[0],v[1]+L[1]))
    c.setFill(color_rgb(color[0], color[1], color[2]))
    #if not occupied:
    #    c.setOutline("white")
    c.draw(win)
def drawRightSideUpTriangle(v,occupied,L,win,color):
    c = Polygon(Point((v[0]-L*0.5),(v[1]+L*np.sqrt(3)/6.0)), Point((v[0]+L*0.5),(v[1])+L*np.sqrt(3)/6.0),Point(v[0],(v[1]-L*np.sqrt(3)/3.0)))
    c.setFill(color_rgb(color[0], color[1], color[2]))
    #if not occupied:
    #    c.setOutline("white")
    c.draw(win)
def drawDownSideUpTriangle(v,occupied,L,win,color):
    c = Polygon(Point((v[0]-L*0.5),(v[1]-L*np.sqrt(3)/6.0)), Point((v[0]+L*0.5),(v[1])-L*np.sqrt(3)/6.0),Point(v[0],(v[1]+L*np.sqrt(3)/3.0)))
    c.setFill(color_rgb(color[0], color[1], color[2]))
    #if not occupied:
    #    c.setOutline("white")
    c.draw(win)
def statistics(clusters):
    Nclusters   = np.max(clusters)
    clusterSize = []
    minSize     = 0
    for i in np.linspace(1,Nclusters,Nclusters):
        x=np.where(clusters==i)
        if len(x[0])>minSize:
            clusterSize=clusterSize+[len(x[0])]
    return clusterSize
    '''
    print(clusterSize)
    n,bins,junk=plt.hist(clusterSize,bins=100,log=True)
    plt.figure()
    plt.plot(bins[:-1],n)
    plt.show()
    '''
def labelCluster(cluster,v,L,win):
    t = Text(Point(L*(v[0]+0.5),L*(v[1]+0.5)), str(cluster))
    t.draw(win)

def mainSquareGrid():
    L,p=5,0.59
    Nx,Ny=int(1000/L),int(750/L)
    #np.random.seed(14)
    probScan=[p]#np.linspace(0,1,100)
    NclustersAtP=[]
    win=0
    for p in probScan:
        print(p)
        V=np.zeros((Nx+2,Ny+2))
        for i in range(Nx):
            for j in range(Ny):
                if np.random.random()<p:
                    V[i+1,j+1]=1
        clusters,classes=findClusters(V,win)
        clusterSizes=statistics(clusters)
        NclustersAtP=NclustersAtP+[len(clusterSizes)]
    plt.figure()
    plt.plot(probScan,NclustersAtP)
    plt.show()
    if 1==1:
        win = GraphWin("Forest Patches", Nx*L, Ny*L)
        win.setBackground('white')
        for i in range(Nx):
            for j in range(Ny):
                drawRect(np.array([i,j]),V[i+1,j+1],L,win,clusterColor(clusters[i+1,j+1]))
    #            labelCluster(clusters[i+1,j+1],np.array([i,j]),L,win)
    win.getMouse() # Pause to view result
    win.close()    # Close window when done


def mainHexgonalGrid():
    L,p=10,0.55
    Nx,Ny=int(1000/L),int(700/L)
    #np.random.seed(14)
    probScan=[p]#np.linspace(0,1,100)
    NclustersAtP=[]
    win=0
    for p in probScan:
        O=np.zeros((Nx+2,Ny+2))
        for i in range(Nx):
            for j in range(Ny):
                if np.random.random()<p:
                    O[i+1,j+1]=1
          #        clusters,classes=findClusters(V,win)

    if 1==1:
        win = GraphWin("Forest Patches", Nx*L, Ny*L)
        win.setBackground('white')
        v=np.array([L,L])
        Oorigin = np.array([L/np.sqrt(3),L/2.0])
        shiftDown  = np.array([0,L])
        shiftDiagDown  = np.array([L*np.sqrt(3)/2.0,L/2.0])
        shiftDiagUp  = np.array([L*np.sqrt(3)/2.0,-L/2.0])
        
        for i in range(Nx):
            for j in range(Ny):
                if np.mod(i,2)==0:#even
                    drawHexagon(Oorigin+i/2*shiftDiagDown+i/2*shiftDiagUp+j*shiftDown,O[i+1,j+1],L,win,clusterColor(O[i+1,j+1]))
                if np.mod(i,2)==1:#odd
                    drawHexagon(Oorigin+(i+1)/2*shiftDiagDown+(i-1)/2*shiftDiagUp+j*shiftDown,O[i+1,j+1],L,win,clusterColor(O[i+1,j+1]))
                
    #            labelCluster(clusters[i+1,j+1],np.array([i,j]),L,win)
    win.getMouse() # Pause to view result
    win.close()    # Close window when done


def mainTriangleGrid():
    L,p=10,0.6
    Nx,Ny=int(1000/L*2),int(700/L*2)
    #np.random.seed(14)
    probScan=[p]#np.linspace(0,1,100)
    NclustersAtP=[]
    win=0
    for p in probScan:
        O=np.zeros((Nx+2,Ny+2))
        X=np.zeros((Nx+2,Ny+2))
        for i in range(Nx):
            for j in range(Ny):
                if np.random.random()<p:
                    O[i+1,j+1]=1
                if np.random.random()<p:
                    X[i+1,j+1]=1
#        clusters,classes=findClusters(V,win)

    if 1==1:
        win = GraphWin("Forest Patches", Nx*L/2.0, Ny*L/2.0)
        win.setBackground('white')
        v=np.array([L,L])
        Oorigin = np.array([L-Nx*L/2,L*np.sqrt(3.0)/3.0])
        Xorigin = np.array([L/2.0-Nx*L/2,L*np.sqrt(3.0)/6.0])
        shiftH  = np.array([L,0])
        shiftD  = np.array([L/2.0,L*np.sqrt(3.0)/2.0])
        for i in range(Nx):
            for j in range(Ny):
                drawRightSideUpTriangle(Oorigin+i*shiftH+j*shiftD,O[i+1,j+1],L,win,clusterColor(O[i+1,j+1]))
                drawDownSideUpTriangle(Xorigin+i*shiftH+j*shiftD,X[i+1,j+1],L,win,clusterColor(X[i+1,j+1]))
    #           
    #            labelCluster(clusters[i+1,j+1],np.array([i,j]),L,win)
    win.getMouse() # Pause to view result
    win.close()    # Close window when done
mainHexgonalGrid()
#mainTriangleGrid()
#mainSquareGrid()


