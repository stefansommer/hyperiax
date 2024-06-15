# Functions for plotting data and tree
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import patches as mpatch
import plotly.graph_objects as go

#####################################################################################################
# 2d and 3d plot of data points in a 

def plot_tree_2d_(self, ax=None, selector=None):
    from matplotlib import pyplot as plt
    from matplotlib import patches as mpatch

    tree = 'partial'
    for node in self.iter_bfs():
        if node.data == None: break
    else:
        tree = 'full'

    cmap = plt.cm.ocean

    if ax == None:
        fig,ax = plt.subplots(figsize=(10,8))
    if tree == 'full':
        levels = list(self.iter_levels())

        for i, level in enumerate(levels):
            for node in level:
                dat = selector(node.data) if selector else node.data
                dat = dat if len(dat.shape) == 1 else dat[-1] # possibly to last value if dat is a trajectory of values
                if node.children:
                    for child in node.children:
                        cdat = selector(child.data) if selector else child.data
                        if len(cdat.shape) == 1:
                            ax.arrow(*dat, *(cdat-dat), width=0.01, length_includes_head=True, color='gray')
                        elif len(cdat.shape) == 2:
                            ax.plot(cdat[:,0], cdat[:,1], color=cmap(i/len(levels)))
                ax.scatter(*dat, color=cmap(i/len(levels)))
                if 'name' in node.data.keys():
                    ax.annotate(node.data['name'], dat, xytext=(5,5), textcoords='offset pixels')

        handles = [mpatch.Patch(color=cmap(i/len(levels)), label = f'{i+1}') for i in range(len(levels))]
        legend = ax.legend(handles=handles, title="Levels")
        ax.add_artist(legend)
        ax.grid(True)

def plot_tree_3d_(self, ax=None, selector=None):

    tree = 'partial'
    for node in self.iter_bfs():
        if node.data == None: break
    else:
        tree = 'full'

    cmap = plt.cm.ocean

    if ax == None:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
    if tree == 'full':
        levels = list(self.iter_levels())

        for i, level in enumerate(levels):
            for node in level:
                dat = selector(node.data) if selector else node.data
                if node.children:
                    for child in node.children:
                        cdat = selector(child.data) if selector else child.data
                        ax.quiver(*dat, *(cdat-dat), length=1.0, arrow_length_ratio=0.1, color='gray')
                ax.scatter(*dat, color=cmap(i/len(levels)))
                if 'name' in node.data.keys():
                    ax.text(*dat, node.data['name'], color='black')

        handles = [mpatch.Patch(color=cmap(i/len(levels)), label=f'{i+1}') for i in range(len(levels))]
        legend = ax.legend(handles=handles, title="Levels")
        ax.add_artist(legend)
        ax.grid(True)

#####################################################################################################
# tree illustration 
        
def estimate_position(self):
    """ Estimate the x and y coordinates of each point """
    
    # Determine Y coordinates, with distance from root 
    self.root.data["y_temp"] = 0
    for leaf in self.iter_dfs():
        # initlize an empty x coordinate for later 
        leaf.data["x_temp"] = 0
        if leaf.parent is not None: # Skip root
            if 'edge_length' in leaf.data.keys():
                leaf.data["y_temp"] = leaf.parent.data["y_temp"] -leaf.data["edge_length"] 

            else: 
                leaf.data["y_temp"] = leaf.parent.data["y_temp"] - 1
    # Define x coordinate for each leaf 
    for i,leaf in enumerate(self.iter_leaves_dfs()):
        leaf.data["x_temp"] = i


    # Determine X coordinates from bottom and up 
    for level in reversed(list(self.iter_levels())):
        for leaf in level:
                while leaf.parent is not None:
                    x_coordinate = 0
                    leaf = leaf.parent
                    for i,node in enumerate(leaf.children):
                        x_coordinate += node.data["x_temp"]
                    leaf.data["x_temp"] = x_coordinate/(i+1)
    return self 
     
def plot_tree_(self,ax=None,inc_names=False): 
    from matplotlib import pyplot as plt
    """Plot the tree using matplotlib"""
 
    if ax == None:
        fig,ax = plt.subplots(figsize=(10,8))

    self = estimate_position(self)
    ax.plot(self.root.data["x_temp"], self.root.data["y_temp"], 'ko')  # Plot the current node
    ax.axis('off')
    for leaf in self.iter_bfs():
        if len(leaf.children) != 0:
            plot_node(leaf,ax,inc_names)

def plot_node(parent,ax,inc_names):
        from matplotlib import pyplot as plt
        """Plot a single node and its children"""
        ax.plot(parent.data["x_temp"], parent.data["y_temp"], 'ko')  # Plot the current node
        if inc_names and parent.name is not None:
            ax.text(parent.data["x_temp"], parent.data["y_temp"], parent.name+" ", fontdict=None,rotation="vertical",va="top",ha="center")
            # Draw vertical line from parent to current level
        for child in parent.children:
            ax.plot(child.data["x_temp"], child.data["y_temp"], 'ko')

            # Include text 
            if inc_names and child.name is not None:
                ax.text(child.data["x_temp"], child.data["y_temp"], child.name+" ", fontdict=None,rotation="vertical",va="top",ha="center")
            # Draw vertical line from parent to current level
            ax.plot([child.data["x_temp"], parent.data["x_temp"]], [parent.data["y_temp"], parent.data["y_temp"]], 'k-')
            # Draw horizontal line to child
            ax.plot([child.data["x_temp"], child.data["x_temp"]], [parent.data["y_temp"], child.data["y_temp"]], 'k-') 


#####################################################################################################
# plot shape tree

def plot_tree_shape(self,ax=None,inc_names=False,shape="landmarks"): 
    from matplotlib import pyplot as plt
    """Plot the tree using matplotlib"""
 
    if ax == None:
        fig,ax = plt.subplots(figsize=(16,10))
        ax.axis('off')

    self = estimate_position_shape(self)
   

    n_leafs = len(list(self.iter_leaves()))
    scale = 7/8
    dis = 1/n_leafs*1/2*scale
    #print(dis)

    ####### DO all for root 
    leaf = self.root

    
    x = leaf.data["x_temp"]
    y =  leaf.data["y_temp"]
    
    # Include text
    if inc_names and leaf.name is not None:
        rotation = "horizontal" if len(leaf.name) < 3 else "vertical"
        ax.text(x, y-dis, leaf.name, fontdict=None, rotation=rotation, va="top", ha="center")


    points = scale_points(leaf.data[shape].reshape((-1,2)),[(x-dis,y-dis),(x+dis,y+dis)])
    for point in points:
        ax.plot(*point, 'ro')
    leaf.data['temp_plotted_point'] = np.array(points)
        
    plot_trajectory = len(leaf.data[shape]) > 1
    if not plot_trajectory:
        draw_box(ax, x, y, dis) # regular case (no trajectory)

    # ax.axis('off')
    n_levels = len(list(self.iter_levels()))
    for i, level in enumerate(self.iter_levels()):
        for node in level:
            if len(node.children) != 0:
                plot_node_shape(node,ax,inc_names,dis,shape,i/n_levels)

    cmap = plt.cm.ocean
    handles = [mpatch.Patch(color=cmap(i/n_levels), label = f'{i+1}') for i in range(n_levels)]
    legend = ax.legend(handles=handles, title="Levels")
    ax.add_artist(legend)

def plot_tree_shape_3d(self,fig=None,inc_names=False,shape="landmarks",scale=1.,mesh=None): 
    import plotly.graph_objects as go
    """Plot the tree using plotly"""
 
    if fig == None:
        fig = go.Figure()
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False), 
                yaxis=dict(visible=False), 
                zaxis=dict(visible=False)
            ),
            showlegend=False,  # Disable legends
            #width = 1000,
            height = 800,
        )

    # simple placement in xz plane, does not take edge lengths into account
    x_span = 1.5*len(list(self.iter_leaves()))
    n_levels = len(list(self.iter_levels()))
    level_z = 2.
    def set_pos(node,childi,nchildren,x_span):
        if node.parent is not None:
            x = node.parent.data['p_temp'][0] + (childi-(nchildren-1)/2)*x_span
            y = 0.
            z = node.parent.data['p_temp'][2] - level_z
        else:
            x = 0.; y = 0.; z = 0.
        node.data['p_temp'] = np.array([x,y,z])
        for i,child in enumerate(node.children):
            set_pos(child,i,len(node.children),x_span/len(node.children))
    tree = self
    set_pos(tree.root,0,len(tree.root.children),x_span)

    ####### DO all for root 
    points = scale*tree.root.data[shape].reshape((-1,3))
    if mesh is None:
        fig.add_trace(go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers', marker=dict(color='blue')))
    else:
        plot_mesh_plotly(mesh,points,fig)
    tree.root.data['temp_plotted_point'] = points
        
    n_levels = len(list(tree.iter_levels()))
    for i, level in enumerate(tree.iter_levels()):
        for node in level:
            if len(node.children) != 0:
                plot_node_shape_3d(node,fig,shape,scale,i/n_levels,mesh=mesh)
    fig.update_layout(scene_aspectmode='data')
    return fig


def estimate_position_shape(self):
    """ Estimate the x and y coordinates of each point """
    
    # Determine Y coordinates, with distance from root 
    self.root.data["y_temp"] = 0
    for leaf in self.iter_dfs():
        # initlize an empty x coordinate for later 
        leaf.data["x_temp"] = 0
        if leaf.parent is not None: # Skip root
            leaf.data["y_temp"] = leaf.parent.data["y_temp"] - leaf.data.get('edge_length', 1)

    # Define x coordinate for each leaf 
    for i,leaf in enumerate(self.iter_leaves_dfs()):
        leaf.data["x_temp"] = i


    # Determine X coordinates from bottom and up 
    for level in reversed(list(self.iter_levels())):
        for leaf in level:
                while leaf.parent is not None:
                    x_coordinate = 0
                    leaf = leaf.parent
                    for i,node in enumerate(leaf.children):
                        x_coordinate += node.data["x_temp"]
                    leaf.data["x_temp"] = x_coordinate/(i+1)


    
    # get the last leaf to see coordinates
    min_depth = 0 ;  min_width = 0
    # Get last leaf in leaves
    leaf = list(self.iter_leaves())[-1]
    max_depth = leaf.data["y_temp"]
    max_width = leaf.data["x_temp"]
    
    # NOrmalize everthing
    try:
        for leaf in self.iter_dfs():
            leaf.data["y_temp"] =- (leaf.data["y_temp"] - min_depth)/(max_depth - min_depth)
            leaf.data["x_temp"] = (leaf.data["x_temp"] - min_width)/(max_width - min_width)
    except ZeroDivisionError:
        pass

    #self.root.data["x_temp"] = 0.5
    return self 

def scale_points(points, bounding_box, padding=0.1):
    # Unpack bounding box coordinates
    box_min_x, box_min_y = bounding_box[0]
    box_max_x, box_max_y = bounding_box[1]

    # Calculate the range of the bounding box
    box_range_x = box_max_x - box_min_x
    box_range_y = box_max_y - box_min_y

    # Add padding to the bounding box
    box_min_x += box_range_x * padding
    box_max_x -= box_range_x * padding
    box_min_y += box_range_y * padding
    box_max_y -= box_range_y * padding
    box_range_x = box_max_x - box_min_x
    box_range_y = box_max_y - box_min_y

    # Find the min and max x, y in the points
    points=np.array(points)
    min_x,min_y=np.min(points,axis=0)
    max_x,max_y=np.max(points,axis=0)

    # Calculate the range of the points
    range_x = max_x - min_x
    range_y = max_y - min_y

    # Scale the points to fit inside the bounding box
    min_x,min_y = np.min(points,axis=0)
    max_x,max_y = np.max(points,axis=0)
    range_x = max_x-min_x
    range_y = max_y-min_y

    # Avoid division by zero
    scaled_x = box_min_x+((points[:,0]-min_x)/range_x)*box_range_x
    scaled_y = box_min_y+((points[:,1]-min_y)/range_y)*box_range_y

    return np.column_stack((scaled_x,scaled_y))


def draw_box(ax, x, y, dis):
    #ax.plot([x-dis,x+dis],[y+dis,y+dis], 'k-')  # Upper Horizontal
    #ax.plot([x-dis,x+dis],[y-dis,y-dis], 'k-')  # Lower horizontal
    #ax.plot([x-dis,x-dis],[y-dis,y+dis], 'k-')  # Vertical lines
    #ax.plot([x+dis,x+dis],[y-dis,y+dis], 'k-')  # Vertical lines opposite
    ax.fill([x-dis, x+dis, x+dis, x-dis], 
            [y-dis, y-dis, y+dis, y+dis], color='white', edgecolor='black')
    


def plot_node_shape(parent, ax, inc_names, dis,shape,level):
    from matplotlib import pyplot as plt

    x0 = parent.data["x_temp"]
    y0 = parent.data["y_temp"]

    cmap = plt.cm.ocean

    for child in parent.children:
        x = child.data["x_temp"]
        y = child.data["y_temp"]

        # plot just point configuration of entire trajectory
        plot_trajectory = len(child.data[shape].shape) > 1

        if not plot_trajectory:
            # Draw horizontal and vertical lines
            if len(parent.children) > 1:
                if x<x0-.5*dis or x>x0+.5*dis:
                    ax.plot([x,x0-dis if x<x0 else x0+dis], [y0,y0],'k-')
                ax.plot([x,x],[y0 if x<x0-.5*dis or x>x0+.5*dis else y0-dis,y+dis],'k')      
            else:
                ax.plot([x,x],[y0-dis,y+dis],'k')

            # Draw box for shape
            draw_box(ax, x, y, dis)

            # Plot points
            points = scale_points(child.data[shape].reshape((-1,2)),[(x-dis,y-dis),(x+dis,y+dis)])
            ax.scatter(points[:,0], points[:,1],color='r',marker='.')
            
            child.data['temp_plotted_point'] = points
        else:
            # Plot points
            points = scale_points(child.data[shape].reshape((-1,2)),[(x-dis,y-dis),(x+dis,y+dis)]).reshape((child.data[shape].shape[0],-1,2))
            if 'temp_plotted_point' in parent.data:
                # linearly interpolate
                child_first_point = points[0]
                child_last_point = points[-1]
                parent_last_point =parent.data['temp_plotted_point']
                num_points = child.data[shape].shape[0]
                interpolated_array = np.linspace(parent_last_point-child_first_point, np.zeros_like(child_last_point), num_points)
                points = points+interpolated_array
            for i in range(points.shape[1]):
                ax.plot(points[:,i,0], points[:,i,1],color=cmap(level))
                ax.plot(*points[-1,i], 'ro')
            child.data['temp_plotted_point'] = points[-1]

        # Include text
        if inc_names and child.name is not None:
            rotation = "horizontal" if len(child.name) < 3 else "vertical"
            ax.text(x, y-dis, child.name, fontdict=None, rotation=rotation, va="top", ha="center")


def plot_node_shape_3d(parent,fig,shape,scale,level,mesh=None):
    cmap = plt.cm.ocean

    for child in parent.children:
        p = child.data['p_temp']

        # plot just point configuration of entire trajectory
        plot_trajectory = len(child.data[shape].shape) > 1

        if not plot_trajectory:
            # no trajectory
            point = scale*child.data[shape].reshape((-1,3)) + p[None,:]
            if 'temp_plotted_point' in parent.data:
                points = np.vstack((parent.data['temp_plotted_point'],point)).reshape([2,-1,3])
            else:
                points = point.reshape([1,-1,3])
        else:
            # trajectory
            points = scale*child.data[shape].reshape((child.data[shape].shape[0],-1,3)) + p[None,None,:]
            if 'temp_plotted_point' in parent.data:
                # linearly interpolate
                child_first_point = points[0]
                child_last_point = points[-1]
                parent_last_point = parent.data['temp_plotted_point']
                num_points = child.data[shape].shape[0]
                interpolated_array = np.linspace(parent_last_point-child_first_point, np.zeros_like(child_last_point), num_points)
                points = points+interpolated_array

        for i in range(points.shape[1]):
            fig.add_trace(go.Scatter3d(x=points[:,i,0], y=points[:,i,1], z=points[:,i,2], mode='lines', line=dict(color=cmap(level))))
            if mesh is None:
                fig.add_trace(go.Scatter3d(x=[points[-1,i,0]], y=[points[-1,i,1]], z=[points[-1,i,2]], mode='markers', marker=dict(color='red', size=6)))
        if mesh is not None:
            plot_mesh_plotly(mesh,points[-1],fig,edgecolor='red')
        child.data['temp_plotted_point'] = points[-1]

def plot_mesh_plotly(mesh,points,fig,color='lightblue',edgecolor='blue'):
    fig.add_trace(go.Mesh3d(x=points[:,0],y=points[:,1],z=points[:,2],i=mesh.faces[:,0],j=mesh.faces[:,1],k=mesh.faces[:,2],color=color,opacity=0.95))
    edges = np.vstack([mesh.faces[:,[0,1]],mesh.faces[:,[1,2]],mesh.faces[:,[2,0]]])
    x_edges = np.hstack([points[edges[:,0],0],points[edges[:,1],0],np.full(edges.shape[0],None)])
    y_edges = np.hstack([points[edges[:,0],1],points[edges[:,1],1],np.full(edges.shape[0],None)])
    z_edges = np.hstack([points[edges[:,0],2],points[edges[:,1],2],np.full(edges.shape[0],None)])
    fig.add_trace(go.Scatter3d(x=x_edges,y=y_edges,z=z_edges,mode='lines',line=dict(color=edgecolor,width=2)))