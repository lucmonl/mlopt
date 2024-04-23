class graphs:
    def __init__(self):
        self.log_epochs   = []
        self.accuracy     = []
        self.loss         = []
        self.eigs         = []
        self.gn_eigs         = []
        self.eigs_test    = []
        self.adv_eigs     = {} # the keys are adv_eta

        self.test_loss    = []
        self.test_accuracy = []

        self.reg_loss     = []
        self.test_reg_loss     = []

        # NC1
        self.Sw_invSb     = []

        # NC2
        self.norm_M_CoV   = []
        self.norm_W_CoV   = []
        self.cos_M        = []
        self.cos_W        = []

        # NC3
        self.W_M_dist     = []

        # NC4
        self.NCC_mismatch = []

        # Decomposition
        self.MSE_wd_features = []
        self.LNC1 = []
        self.LNC23 = []
        self.Lperp = []

        # NC1
        self.test_Sw_invSb     = []

        # NC2
        self.test_norm_M_CoV   = []
        self.test_norm_W_CoV   = []
        self.test_cos_M        = []
        self.test_cos_W        = []

        # NC3
        self.test_W_M_dist     = []

        # NC4
        self.test_NCC_mismatch = []

        # Decomposition
        self.test_MSE_wd_features = []
        self.test_LNC1 =         []
        self.test_LNC23 =        []
        self.test_Lperp =        []

        # weight norm statsitics
        self.wn_grad_loss_ratio = []
        self.wn_norm_min        = []
        self.wn_norm_min_with_g = []

        # weight alignment
        self.align_signal_1     = []
        self.align_signal_2     = []
        self.align_noise        = []
        self.linear_coefs       = []
        self.model_output       = []
        self.activation_pattern = []

        # diagonal statistics
        self.diagonal_coef      = []
        self.diagonal_invariate = []

        self.cos_descent_ascent = []
