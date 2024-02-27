import numpy as np
import logging

Sw = 1
cw = 0.038
cp = 7.5e-12
l = 1 / 0.09


def ceff(m, cw=cw, cp=cp, l=l, Sw=Sw):

    F = 1 + np.exp(-m[:int(len(m) / 2)])
    spor = np.exp(m[int(len(m) / 2):])

    return 1 / F * cw * Sw**2 + l * cp * spor + 1j * cp * spor


def dcrF(m, cw=cw, cp=cp, l=l, Sw=Sw):

    F = 1 + np.exp(-m[:int(len(m) / 2)])
    spor = np.exp(m[int(len(m) / 2):])

    return - F ** (-2) * cw * Sw**2 * -np.exp(-m[:int(len(m) / 2)])


def dcrSpor(m, cw=cw, cp=cp, l=l, Sw=Sw):

    F = 1 + np.exp(-m[:int(len(m) / 2)])
    spor = np.exp(m[int(len(m) / 2):])

    return l * cp * spor


def dciF(m, cw=cw, cp=cp, l=l, Sw=Sw):

    F = 1 + np.exp(-m[:int(len(m) / 2)])
    spor = np.exp(m[int(len(m) / 2):])

    return 0


def dciSpor(m, cw=cw, cp=cp, l=l, Sw=Sw):

    F = 1 + np.exp(-m[:int(len(m) / 2)])
    spor = np.exp(m[int(len(m) / 2):])

    return cp * spor


def rhoeff(m, cw=cw, cp=cp, l=l, Sw=Sw):

    return 1 / ceff(m, cw=cw, cp=cp, l=l, Sw=Sw)


def line_search_eta(rmse, m0, dm, eta, *rmse_params):

    etas = np.array([0, 0.5, 1]) * eta
    rmses = np.zeros(3)

    for ieta, eta in enumerate(etas):

        m = m0 + eta * dm
        rmses[ieta] = rmse(m, *rmse_params)


    A = np.zeros((3, 3))
    A[:, 0] = etas**2
    A[:, 1] = etas
    A[:, 2] = 1

    a, b, c = np.linalg.solve(A, rmses)

    etamin = - b / (2 * a)

    if etamin > 0 and etamin <= 1:
        return etamin

    return 1


def check_conjugate(x):

    if not np.allclose(x[:int(len(x) / 2)], x[int(len(x) / 2):].conj()):
        raise ConjugateError("The model does not fullfil the Conjugate Coordinates.")


class Complex_Inversion_Manager:

    def __init__(self, m, d, solve_forward_problem, lam, Rd, Rm, mp):

        self.m = m
        self.check_conjugate(self.m)

        self.d = d
        self.check_conjugate(self.d)

        self.mp = mp

        self.M = int(len(self.m) / 2)
        self.D = int(len(self.d) / 2)

        self.solve_forward_problem = solve_forward_problem

        self.Rd = Rd
        self.Rm = Rm
        self.Jc = None
        self.response = None

        self.J_is_conj_coo = 0

        self.invVr = np.eye(self.D)
        self.invVi = np.eye(self.D)

        self._S = np.zeros((2 * self.M, 2 * self.M))
        self._S[:self.M, self.M:] = np.eye(self.M)
        self._S[self.M:, :self.M] = np.eye(self.M)

        self.lam = lam
        self.eta = 1
        self.threshold_norm = 1e-4
        self.cost = np.infty


    def calculate_cost(self, f, m):

        return (f - self.d).conj().T @ self.Rd @ (f - self.d) + self.lam * (m - self.mp).conj().T @ self.Rm @ (m - self.mp)


    def check_conjugate(self, x):

        if not np.allclose(x[:int(len(x) / 2)], x[int(len(x) / 2):].conj()):
            raise ConjugateError("The model does not fullfil the Conjugate Coordinates.")


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

        f, J = self.solve_forward_problem(m)

        self.response = f

        _rmse = self.rmse(self.d, f)

        print("rmse ", _rmse)

        # if _rmse[0] < 1.1:
        #     perform_line_search = 1
        # if _rmse[1] < 1.1:
        #     perform_line_search = 1
        # if _rmse[2] < 1.1:
        #     perform_line_search = 1

        if self.J_is_conj_coo:
            self.Jc = J
        else:
            self.Jc = np.vstack([
                np.hstack([J, np.zeros_like(J)]),
                np.hstack([J, np.zeros_like(J)]).conj() @ self._S
            ])

        B = self.Jc.conj().T @ self.Rd @ self.Jc + self.lam * self.Rm
        Binv = np.linalg.inv(B)

        b = self.Jc.conj().T @ self.Rd @ (self.d - f) - self.lam * self.Rm @ (self.m - self.mp)

        update = Binv @ b
        norm = np.linalg.norm(update) / (2 * self.M)

        if (not perform_line_search):

            if (norm < self.threshold_norm):
                return update, True

            return update, False

        # line search

        etas = np.array([0, 0.5, 1]) * self.eta
        rmses = np.zeros(3)

        rmses[0] = self.rmse(self.d, f)[0]

        for ieta, eta in enumerate(etas[1:], 1):

            f, _ = self.solve_forward_problem(self.m + eta * update)

            rmses[ieta] = 0.5 * ( (self.d - f).conj().T @ self.Rd @ (self.d - f)
                    + self.lam * (self.m - self.mp).conj().T @ self.Rm @ (self.m - self.mp) ).real

        A = np.zeros((3, 3))
        A[:, 0] = etas**2
        A[:, 1] = etas
        A[:, 2] = 1

        a, b, c = np.linalg.solve(A, rmses)

        self.eta = - b / (2 * a)

        print("eta ", self.eta)

        if not (self.eta > 0 and self.eta <= 1):
            self.eta = 1

        if norm * self.eta < self.threshold_norm:
            return update, True

        return update, False


    def invert(self, N_iterations=10, perform_line_search=True, ignore_cost_increase=0):

        for i in range(N_iterations):

            logging.info("Iteration {}".format(i))

            update, exit = self.calculate_pseudo_newton_update(self.m, perform_line_search=perform_line_search)

            _cost = self.calculate_cost(self.response, self.m + self.eta * update)

            if( _cost > self.cost) and not ignore_cost_increase:
                logging.info("Cost increased...")
                break

            self.cost = _cost

            if exit:
                break

            self.m = self.m + self.eta * update

            self.check_conjugate(self.m)

            # try:
            #     self.check_conjugate(self.m)
            # except ConjugateError as error:
            #     print("ConjugateError")
            #     logging.info("ConjugateError")

        # else:
        #     raise NotConvergedError("The inversion did not reach a solution during the specified number of N_iterations.")


class RTO_Chain:

    def __init__(self,
                starting_model,
                map_solution,
                map_joined_jacobian,
                dimension_model_space,
                chain_steps,
                burn_in_steps,
                propose_update,
                name="RTO_Chain",
                output_path=None
                ):

        self.name = name
        if output_path == None:
            self.output_path = self.name
        else:
            self.output_path = output_path

        self.dimension_model_space = dimension_model_space
        self.chain_steps = chain_steps
        self.burn_in_steps = burn_in_steps

        self.current_model = starting_model
        self.current_lnc = 0
        self.iteration = 0
        self.updates_accepted = 0

        self.sample_array = np.zeros((self.dimension_model_space, self.chain_steps)).astype(complex)

        self.map_solution = map_solution
        self.map_joined_jacobian = map_joined_jacobian
        self.Q = np.linalg.qr(self.map_joined_jacobian)[0]

        # Functions
        self.propose_update = propose_update


    def run(self, info_interval=1000, checkpoint_interval=1000, print_info=0):

        proposed_model, proposed_lnc = self.propose_update()

        self.current_model = np.copy(proposed_model)
        self.current_lnc = proposed_lnc

        # self.iteration += 1
        self.sample_array[:, 0] = self.current_model

        for i in range(self.iteration+1, self.chain_steps):

            proposed_model, proposed_lnc = self.propose_update()

            alpha = (self.current_lnc - proposed_lnc)
            u = np.log(np.random.uniform(low=0, high=1, size=1)[0])

            print(alpha)

            if (u < alpha):
                self.current_model = np.copy(proposed_model)
                self.current_lnc = proposed_lnc

            self.iteration += 1
            self.sample_array[:, i] = self.current_model

            self.display_info(info_interval, print_info)
            self.check_checkpoint(checkpoint_interval)



    def create_checkpoint(self, filename=None):
        if filename == None:
            filename = self.output_path
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")

        filehandler = open("checkpoints/" + str(filename) + ".pickle", "wb")
        pickle.dump(self, filehandler)

        logging.info("Create checkpoint of rto chain {} as iteration {}".format(self.name, self.iteration))


    def check_checkpoint(self, interval=None):
        if interval == None:
            pass
        elif self.iteration % interval == 0:
            self.create_checkpoint()


    def calculate_acceptance_ratio(self):
        return self.updates_accepted / self.iteration


    def display_info(self, interval=1000, print_info=True):
        if self.iteration % interval == 0:
            try:
                if print_info: raise ValueError
                logging.info("{} at {}/{} steps. Acceptance ration: {}".format(self.name,
                    self.iteration, self.chain_steps, self.calculate_acceptance_ratio()))
            except:
                print("{} at {}/{} steps. Acceptance ration: {}".format(self.name,
                    self.iteration, self.chain_steps, self.calculate_acceptance_ratio()))


class ConjugateError(Exception):
    pass


class NotConvergedError(Exception):
    pass
