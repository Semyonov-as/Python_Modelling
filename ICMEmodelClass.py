import scipy.optimize as opt
import numpy as np

class ICMEmodel:
    def __init__(self, width, n1, n2, n3) -> None:
        self.width = width
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.g1 = 107145
        self.g2 = 59525
        self.g3 = 14880/2 #because in china results was just g3, but we use 2*g3

        self.g1 /= self.g3
        self.g2 /= self.g3
        self.g3 /= self.g3

# mode eqs

    def TE_mode_eq(self, k, beta, N=0):
        width, n1, n2, n3 = self.width, self.n1, self.n2, self.n3
        p = np.sqrt(beta**2 - (k*n1)**2)
        q = np.sqrt(-beta**2 + (k*n2)**2)
        r = np.sqrt(beta**2 - (k*n3)**2)
        tmp = np.arctan(q*(p+r)/(q*q - p*r))
        if (q*q - p*r) < 0:
            tmp += np.pi
            
        return np.real(width*q - np.pi*(N) - tmp)

    def TM_mode_eq(self, k, beta, N=0):
        width, n1, n2, n3 = self.width, self.n1, self.n2, self.n3
        p = np.sqrt(beta**2 - (k*n1)**2)
        q = np.sqrt(-beta**2 + (k*n2)**2)
        r = np.sqrt(beta**2 - (k*n3)**2)
        return np.real(width*q - np.arctan(n2**2*p/n1**2/q) - np.arctan(n2**2*r/n3**2/q) - np.pi*N)

    # get solution for selected lambda

    def sol_mode_eq(self, lam, mode='TE', N=0):
        n1, n2, n3= self.n1, self.n2, self.n3
        mode_eq = self.TE_mode_eq if mode=='TE' else self.TM_mode_eq
        k = 2*np.pi/lam
        n_l = np.max((n1, n3))
        l_b = k*(n_l + 1e-3)
        r_b = k*(n2 - 1e-3)
        
        if mode_eq(k, l_b, N=N)*mode_eq(k, r_b, N=N) < 0:
            return opt.brentq(lambda x: mode_eq(k, x, N=N), l_b, r_b)
        else:
            return None

    def get_E_field_TE(self, beta, lam, z):
        n2, n1 = self.n2, self.n1
        k = 2*np.pi/lam
        p = np.sqrt(beta**2 - (k*n1)**2)
        q = np.sqrt(-beta**2 + (k*n2)**2)
        
        return (0, np.cos(q*z) + p/q*np.sin(q*z), 0)

    def get_E_field_TM(self, beta, lam, z):
        n1, n2 = self.n1, self.n2
        k = 2*np.pi/lam
        p = np.sqrt(beta**2 - (k*n1)**2)
        q = np.sqrt(-beta**2 + (k*n2)**2)

        Ez = beta*(-n1**2*q/(n2**2*p)*np.cos(q*z) - np.sin(q*z))
        Ex = -1j*q*(-n1**2*q/(n2**2*p)*np.sin(q*z) + np.cos(q*z))
        return (Ex, 0, Ez)

    # calculate ICME according to phi 

    def ICME_arbirtary_E(self, E, phi, M):
        g1, g2, g3 = self.g1, self.g2, self.g3
        Ex, Ey, Ez = E
        Excon = np.conjugate(Ex)
        Eycon = np.conjugate(Ey)
        Ezcon = np.conjugate(Ez)
        m1, m2, m3 = M
        cos = np.cos(phi)
        sin = np.sin(phi)

        Hx = 2*np.real(cos**3*(Ex*Excon*g1*m1 + Ey*Eycon*g2*m1 + 
            Excon*Ey*g3*m2 + Ex*Eycon*g3*m2) - cos**2*(-((Excon*Ez +
            Ex*Ezcon)*g3*m3) + (Excon*Ey*(-g1 + g2 + g3)*m1 + 
            Ex*Eycon*(-g1 + g2 + g3)*m1 + Ey*Eycon*(g1 - 2*g3)*m2 + 
            Ex*Excon*(g2 + 2*g3)*m2)*sin) + sin*(-Ez*Ezcon*g2*m2 + 
            (Excon*Ez + Ex*Ezcon)*g3*m3*sin + (Excon*Ey*g3*m1 
            + Ex*Eycon*g3*m1 - Ex*Excon*g1*m2 - Ey*Eycon*g2*m2)*sin**2) + 
            cos*(Ez*Ezcon*g2*m1 + (Ey*Eycon*(g1 - 2*g3)*m1 
            + Ex*Excon*(g2 + 2*g3)*m1 + Excon*Ey*(g1 - g2 - g3)*m2 + 
            Ex*Eycon*(g1 - g2 - g3)*m2)*sin**2))   

        Hy = 2*np.real(cos**3*(Excon*Ey*g3*m1 + Ex*Eycon*g3*m1 + 
            Ey*Eycon*g1*m2 + Ex*Excon*g2*m2) + cos**2*((Eycon*Ez +
            Ey*Ezcon)*g3*m3 + (Ex*Excon*(g1 - 2*g3)*m1 + Ey*Eycon*(g2 
            + 2*g3)*m1 + Excon*Ey*(-g1 + g2 + g3)*m2 + Ex*Eycon*(-g1 
            + g2 + g3)*m2)*sin) + sin*(Ez*Ezcon*g2*m1 + (Eycon*Ez + 
            Ey*Ezcon)*g3*m3*sin + (Ey*Eycon*g1*m1 + Ex*Excon*g2*m1 - 
            Excon*Ey*g3*m2 - Ex*Eycon*g3*m2)*sin**2) + cos*(Ez*Ezcon*g2*m2 + 
            (Excon*Ey*(g1 - g2 - g3)*m1 + Ex*Eycon*(g1 - g2 - g3)*m1 + 
            Ex*Excon*(g1 - 2*g3)*m2 + Ey*Eycon*(g2 + 2*g3)*m2)*sin**2))

        Hz = 2*np.real(g3*cos*(Excon*Ez*m1 + Ex*Ezcon*m1 + 
            (Eycon*Ez + Ey*Ezcon)*m2) + (Ez*Ezcon*g1 + Ex*Excon*g2 + 
            Ey*Eycon*g2)*m3 + g3*(Eycon*Ez*m1 + Ey*Ezcon*m1 - 
            (Excon*Ez + Ex*Ezcon)*m2)*sin)

        return (Hx, Hy, Hz)

    @staticmethod
    def get_H(H, psi, eta):
        h1 = H*np.cos(psi)*np.cos(eta)
        h2 = H*np.sin(psi)*np.cos(eta)
        h3 = H*np.sin(eta)
        return (h1, h2, h3)

    @staticmethod
    def get_E(E, ksi, theta):
        e1 = E*np.cos(ksi)*np.cos(theta)
        e2 = E*np.sin(ksi)*np.cos(theta)
        e3 = E*np.sin(theta)
        return (e1, e2, e3)

    @staticmethod
    def get_M_from_H(H, phi):
        h1, h2, h3 = H
        m1 = h1*np.cos(phi) + h2*np.sin(phi)
        m2 = -h1*np.sin(phi) + h2*np.cos(phi)
        m3 = h3

        return (m1, m2, m3)

    @staticmethod
    def normalize_E(E):
        E1, E2, E3 = E
        norm = np.max(np.sqrt(np.abs(E1)**2 + np.abs(E2)**2 + np.abs(E3)**2))
        return (E1/norm, E2/norm, E3/norm)

    def get_icme_torque(self, E, H, M, phi):
        icme = self.ICME_arbirtary_E(E, phi, M)
        return np.cross(H, np.array(icme).transpose())


    # Spin dynamics code

    def A_n(self, n, lam, beta, N, mode='TE'):
        if not n%2 and mode =='TE': #even
            return self.A_n_TE_even(n, lam, beta, N)
        elif not n%2 and mode == 'TM':
            pass
        elif n%2 and mode == 'TE': #odd
            return self.A_n_TE_odd(n, lam, beta, N)
        elif n%2 and mode == 'TM':
            pass

    def A_n_TE_even(self, n, lam, beta, N):
        kz = np.sqrt((2*np.pi*self.n2/lam)**2 - beta**2)
        alpha_TE = np.pi*N/2
        if self.n1 != self.n3:
            gamma1 = np.sqrt(beta**2 - (2*np.pi*self.n1/lam)**2)
            gamma3 = np.sqrt(beta**2 - (2*np.pi*self.n3/lam)**2)
            alpha_TE += 0.5*np.arctan(kz*(gamma1 - gamma3)/(kz**2 + gamma1*gamma3))
        Kn = n*np.pi/self.width

        return 4*kz*(-1)**(n//2)/(4*kz**2 - Kn**2)*np.sin(kz*self.width)*np.cos(2*alpha_TE)


    def A_n_TE_odd(self, n, lam, beta, N):
        kz = np.sqrt((2*np.pi*self.n2/lam)**2 - beta**2)
        alpha_TE = np.pi*N/2
        if self.n1 != self.n3:
            gamma1 = np.sqrt(beta**2 - (2*np.pi*self.n1/lam)**2)
            gamma3 = np.sqrt(beta**2 - (2*np.pi*self.n3/lam)**2)
            alpha_TE += 0.5*np.arctan(kz*(gamma1 - gamma3)/(kz**2 + gamma1*gamma3))
        Kn = n*np.pi/self.width

        return 4*kz*(-1)**((n-1)//2)/(- 4*kz**2 + Kn**2)*np.sin(kz*self.width)*np.cos(2*alpha_TE)
