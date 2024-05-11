#
# Brief    : Implements a specific configuration of 
#            the CST method presented in the papers:
#
#            1. Kulfan, B. M, Bussoletti, J. E., 
#            “"Fundamental Parametric Geometry 
#            Representations for Aircraft Component
#            Shapes”. AIAA-2006-6948, 11th AIAA/ISSMO 
#            Multidisciplinary Analysis and Optimization 
#            Conference: The Modeling and Simulation Frontier
#            for Multidisciplinary Design Optimization, 6 - 8 
#            September, 2006 
#
#            2. Kulfan, B. M..,“Universal Parametric Geometry 
#            Representation Method – “CST”, AIAA-2007-0062, 
#            45th AIAAAerospace Sciences Meeting and Exhibit 
#            , January, 2007
#
# File     : airfoil_generator.py
# Authors  : Deniz Alper Acar,
# Licenese : MIT
#


from math import factorial, tan, pi
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# Common required mrtthods

def binomial(r:int, n:int) -> int:
    """Computes the binomial of r given n"""
    return factorial(n) / (factorial(r) * factorial(n-r))

def bernstein_polynomial(x:list[float], r:int, n:int) -> np.ndarray:
    """
    Implementation of the general Bernstein polynomial 
    term in the formulation.
    """
    x = np.array(x)
    results = binomial(r,n) * (x**r) * ((1-x)**(n-r))
    return results


class Airfoil:
  """This class uses CST method to generate airfoil shapes.

  The naming of the coefficients are not changed to be 
  consistent with the paper,
  """

  def __init__(
      self,
      Nnose:float=0.5, # class function exponent nose
      Naft:float=1.0,  # class function exponent aft
      n:int=4,
      sampling_points:np.ndarray=np.concatenate(
        (
          np.linspace(0, 0.195, 81), 
          np.linspace(0.2, 1., 101)
          )
        ) ,
      rle:float=0.01,
      bu:float=20.,
      dzu:float=0.002,
      bl:float=20.,
      dzl:float=-0.001,
      au_lim:list[float]=[-0.1, 0.6],
      al_lim:list[float]=[-0.6, 0.1], 
      name:str="",
      show:bool=False
    ):
    self.name = name
    self.au_lim = au_lim
    self.al_lim = al_lim
    self.n = n

    self.au = np.random.uniform(self.au_lim[0], self.au_lim[1], self.n + 1)
    self.al = np.random.uniform(self.al_lim[0], self.al_lim[1], self.n + 1)

    self.sampling_points = sampling_points
    self.Nnose = Nnose
    self.Naft = Naft
    self.rle = rle
    self.bu = bu
    self.bl =bl
    self.dzu = dzu
    self.dzl = dzl
    self.b = [bu, bl]
    self.dz = [dzu, dzl]
    
    self.show = show

    self.generate_airfoil()

  def generate_airfoil(self):

    self.au[0] = (self.rle * 2) ** 0.5
    self.au[-1]= tan(self.bu * pi / 180.) + self.dzu
    self.al[0] = -(self.rle *2)**0.5
    self.al[-1]= tan(self.bl * pi / 180.) + self.dzl

    # the calculation
    c = self._get_class(self.sampling_points, self.Nnose, self.Naft)
    # create upper surface
    # calculate Si
    resu = np.zeros_like(self.sampling_points)
    resl = np.zeros_like(self.sampling_points)

    for i in range(0, self.n + 1):
        resu += (
           self.au[i] 
           * bernstein_polynomial(self.sampling_points, i, self.n)
          )
        resl += ( 
          self.al[i] 
          * bernstein_polynomial(self.sampling_points, i, self.n)
          )

    resu = resu * c + self.sampling_points * self.dzu
    resl = resl * c + self.sampling_points * self.dzl

    self.__dict__['yu'] = resu
    self.__dict__['yl'] = resl

  def write(
      self, file_name:str, update_name=True, 
      save_pickle:bool=True, plot:bool=True
    ) -> None:
    """Write generated airofil to a text file and 
    the class to a pickle file.
    """


    directory = "/".join(file_name.split("/")[:-1])
    if update_name:
      self.name = file_name.split("/")[-1].replace(".af", "")
    os.makedirs(directory, exist_ok = True)
    os.makedirs(directory + '/airfoil/', exist_ok = True)
    x = self.sampling_points
    yu = self.yu
    yl = self.yl

    with open(directory + f'/airfoil/{self.name}.af', 'w') as fid:
      fid.write('index {}\n'.format(self.name))
      for pt in range(x.shape[0]):
        fid.write('{:6.5f} {:6.5f}\n'.format(x[x.shape[0]-pt-1], yu[x.shape[0]-pt-1]))
      for pt in range(1,x.shape[0]):
        fid.write('{:6.5f} {:6.5f}\n'.format(x[pt], yl[pt]))
    
    if save_pickle:
      os.makedirs(directory+'/pickle/', exist_ok = True)
      with open(directory + f'/pickle/{self.name}.pickle', 'wb') as fid:
        pickle.dump(self, fid)

    if plot:
      os.makedirs(directory + '/plot/', exist_ok = True)
      fig, ax = self.plot(self.name)
      fig.savefig(directory + f'/plot/{self.name}.png', dpi=100)
      plt.close(fig)

  def _get_class(self, sampling_points:np.ndarray, N1:float, N2:float):
    return (sampling_points ** N1) * ((1 - sampling_points)**N2)

  def read(self, file_name:str) -> bool:
    """Reads generated airfoil from a file.
    """
    with open(file_name, "rb") as fid:
      self = pickle.load(fid)
    return True
  
  def plot(self, ind:int):
    """Plots the current airfoil
    """
    fig, ax = plt.subplots()
    ax.plot(self.sampling_points, self.yu)
    ax.plot(self.sampling_points, self.yl)
    ax.axis('equal')
    plt.title(ind)
    plt.grid()
    return fig, ax

  def __str__(self) -> str:
    """Write the coordinates in Salig format to the screen.
    """
    s = ""

    x = self.sampling_points
    yu = self.yu
    yl = self.yl

    for pt in range(x.shape[0]):
      s += '{:6.5f} {:6.5f}\n'.format(x[x.shape[0]-pt-1], yu[x.shape[0]-pt-1])
    for pt in range(1,x.shape[0]):
      s += '{:6.5f} {:6.5f}\n'.format(x[pt], yl[pt])
    return s 


class AirfoilGenerator:
  """Generates and stores a number of airfoil shapes.
  """
  def __init__(self, num_airfoils:int=20):
    self.start_index = 0
    self._read_last_index()
    self.num_airfoils = num_airfoils
    self.airfoils = []
    self.generate()

  def _read_last_index(self):
    if not os.path.exists('src/AirfoilGenerator/cfg/index.txt'):
      self.start_index = 0
      print(self.start_index)
    else:
      with open('src/AirfoilGenerator/cfg/index.txt', 'r') as fid:
        self.start_index = int(fid.read())


  def generate(self) -> bool:
    """"Generates a set of airfoil shapes"""
    success = False
    # try:
    cd = os.getcwd() + "/"
    for ind in range(
        self.start_index, 
        self.start_index + self.num_airfoils 
      ):


      n=np.random.randint(2,8)
      rle=np.random.uniform(0.01,0.05)
      bu=np.random.randint(-10,20)
      bl=np.random.randint(-20,bu)
      dzu = np.random.uniform(-0.001, 0.005)
      dzl = np.random.uniform(-0.005, dzu-0.001)
      Nnose = np.random.uniform(0.45,0.55)
      Naft  = np.random.uniform(0.95, 1.0)

      self.airfoils.append(
        Airfoil(
          show=False, n=n, 
          rle=rle, bu=bu, bl=bl, 
          dzl=dzl, dzu=dzu, 
          Naft=Naft, Nnose=Nnose,
          sampling_points=np.concatenate(
            (
              np.linspace(0, 0.195, 61), 
              np.linspace(0.2, 1., 101)
              )
            )
          )
        )

      directory = '{}data/'.format(cd)
      os.makedirs(directory, exist_ok = True)
      self.airfoils[-1].write(directory + f'{ind}.af', plot=False)

      with open('src/AirfoilGenerator/cfg/index.txt', 'w') as fid:
        fid.write(str(ind))


  

