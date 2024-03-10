import numpy as np
import logging

class ConjugateError(Exception):
    pass


def check_conjugate(x):
    """
    Checks if the provided complex vector x fullfills the conjugate coordinate
        representation, which means that x[:int(len(x)/2)] == x[int(len(x)/2):].conj().
    If this is not fullfilled, the functions throws a ConjugateError.
    """
    if not np.allclose(x[:int(len(x) / 2)], x[int(len(x) / 2):].conj()):
        raise ConjugateError("The model does not fullfil the Conjugate Coordinates.")


class Complex_Inversion_Manager:

    """
    This class holds all functionality needed for the complex valued probabilistic inversion.
    """

    def __init__(self, m, d, solve_forward_problem, lam, Rd, Rm, mp):

        """
        m [numpy.ndarray] : start model. Must be in conjugate coordinates.
        d [numpy.ndarray]: data to be inverted. Must be in conjugate coordiantes.
        solve_forward_problem [function] : evaluation of the forward response.
            Must take a model in conjugate coordinates. Must return the response
            in conjugate coordinates and the jacobian (standard complex formulation)
        lam [float] : parameter scaling the prior information. Set to 1 if not used.
        Rd [numpy.ndarray] : inverse data covariance matrix in conjugate coordinates.
            Example for Rd construction:
                V_Re = np.diag(std_real ** 2)
                V_Im = np.diag(std_imag ** 2)

                Rd = np.vstack((
                    np.hstack((V_Re + V_Im, V_Re - V_Im)),
                    np.hstack(((V_Re - V_Im).conj().T, (V_Re + V_Im).conj()))
                ))
                Rd = np.linalg.inv(Rd)
        Rm [numpy.ndarray] : inverse prior covariance matrix in conjugate coordinates.
            Example for Rm contruction:
                alpha = np.exp(6.3)
                beta = np.exp(11)

                VM_on = C_M_inv * (alpha + beta)
                VM_off = C_M_inv * (alpha - beta)

                Rm = np.vstack((
                    np.hstack((VM_on, VM_off)),
                    np.hstack((VM_off.conj().T, VM_on))
                ))
        mp [numpy.ndarray] : prior model. Must be in conjugate coordinates.
        """

        self.m = m
        check_conjugate(self.m)

        self.d = d
        check_conjugate(self.d)

        self.mp = mp

        self.M = int(len(self.m) / 2)
        self.D = int(len(self.d) / 2)

        self.solve_forward_problem = solve_forward_problem

        self.Rd = Rd
        self.Rm = Rm
        self.Jc = None
        self.response = None
        self.J_is_conj_coo = False # It is assumed that solve_forward_problem returns the jacobian in its standard complex form.
                                   # If the jacobian is returned in conjugate coordinates, change this to True after creating the Complex_Inversion_Manager object.

        self.invVr = np.eye(self.D)
        self.invVi = np.eye(self.D)

        self._Q = np.zeros((2 * self.M, 2 * self.M))
        self._Q[:self.M, self.M:] = np.eye(self.M)
        self._Q[self.M:, :self.M] = np.eye(self.M)

        self._S = np.hstack([np.eye(self.M), 1j * np.eye(self.M)])
        self._S = np.vstack([self._S, self._S.conj()])
        self._S_inv = np.linalg.inv(self._S)

        self.lam = lam
        self.eta = 1
        self.threshold_norm = 1e-4
        self.threshold_rmse = 1e-2
        self.current_rmse = np.infty
        self.cost = np.infty


    def calculate_cost(self, f, m):
        return (f - self.d).conj().T @ self.Rd @ (f - self.d) + self.lam * (m - self.mp).conj().T @ self.Rm @ (m - self.mp)


    def rmse(self, d, f):

        rmse_complex = np.sqrt(
            (d - f).conj().T @ self.Rd @ (d - f) / (self.D * 2)
        )
        rmse_real = np.sqrt(
            (d[:int(len(d) / 2)] - f[:int(len(f) / 2)]).real.T @ self.invVr @ (d[:int(len(d) / 2)] - f[:int(len(f) / 2)]).real / self.D
        )
        rmse_imag = np.sqrt(
            (d[:int(len(d) / 2)] - f[:int(len(f) / 2)]).imag.T @ self.invVi @ (d[:int(len(d) / 2)] - f[:int(len(f) / 2)]).imag / self.D
        )

        return np.real(rmse_complex), np.real(rmse_real), np.real(rmse_imag)


    def calculate_pseudo_newton_update(self, m, perform_line_search):

        """
        Calculates and returns the pseudo-newton model update.
        m [numpy.ndarray] : Model to be updated. Must be in conjugate coordiantes.
        perform_line_search [bool] : True if line search for stepsize should be performed.
        """

        f, J = self.solve_forward_problem(m)
        self.response = f

        _rmse = self.rmse(self.d, f)
        print("rmse ", _rmse)

        if self.J_is_conj_coo:
            self.Jc = J
        else:
            self.Jc = np.vstack([
                np.hstack([J, np.zeros_like(J)]),
                np.hstack([J, np.zeros_like(J)]).conj() @ self._Q
            ])

        B = self.Jc.conj().T @ self.Rd @ self.Jc + self.lam * self.Rm
        # Binv = np.linalg.inv(B)
        b = self.Jc.conj().T @ self.Rd @ (self.d - f) - self.lam * self.Rm @ (m - self.mp)
        # update = Binv @ b
        update = np.linalg.solve(B, b) # use solve instead of np.linalg.inv for better performance

        # check if minimum is reached
        norm = np.linalg.norm(update) / (2 * self.M)

        if (not perform_line_search):
            if (norm < self.threshold_norm):
                return update, True
            return update, False

        # line search step length
        etas = np.array([0, 0.5, 1]) * self.eta
        rmses = np.zeros(3)

        rmses[0] = 0.5 * ( (self.d - f).conj().T @ self.Rd @ (self.d - f)
                    + self.lam * (m - self.mp).conj().T @ self.Rm @ (m - self.mp) ).real

        for ieta, eta in enumerate(etas[1:], 1):

            f, _ = self.solve_forward_problem(m + eta * update)
            rmses[ieta] = 0.5 * ( (self.d - f).conj().T @ self.Rd @ (self.d - f)
                    + self.lam * (m - self.mp).conj().T @ self.Rm @ (m - self.mp) ).real

        A = np.zeros((3, 3))
        A[:, 0] = etas**2
        A[:, 1] = etas
        A[:, 2] = 1
        a, b, c = np.linalg.solve(A, rmses)

        self.eta = - b / (2 * a)
        print("eta ", self.eta)

        if not (self.eta > 0):
            self.eta = 1
        if norm * self.eta < self.threshold_norm:
            return update, True
        return update, False


    def inversion(self, N_iterations=10, perform_line_search=True, ignore_cost_increase=False):

        """
        Perform the inversion.
        N_iterations [int] : maximum number of iterations to be performed.
        perform_line_search [bool] : True if line search for stepsize should be performed.
        ignore_cost_increase [bool] : set true if inversion should be stopped if
            cost to be minimized increases momentarily.
        """

        self.rmse0, self.rrmse0, self.irmse0 = self.rmse(self.d, self.solve_forward_problem(self.m)[0])
        print(self.rmse0, self.rrmse0, self.irmse0)

        for i in range(N_iterations):

            logging.info("\n" + "#"*20 + "\nMain iteration {}\n".format(i) + "#" * 20)

            update, stop_optimization = self.calculate_pseudo_newton_update(self.m, perform_line_search=perform_line_search)
            _cost = self.calculate_cost(self.response, self.m + self.eta * update)

            if np.abs(_cost - self.cost) < self.threshold_norm:
                perform_line_search = True

            if( _cost > self.cost) and not ignore_cost_increase:
                logging.info("Cost increased...")

                if perform_line_search:
                    break
                perform_line_search = True
                continue

            self.cost = _cost

            check_conjugate(update)

            if stop_optimization:
                break

            self.m = self.m + self.eta * update

        self.response = self.solve_forward_problem(self.m)[0]
        logging.info("Final rmse: {}".format(self.rmse(self.d, self.response)))
