import torch
from gekko import GEKKO
from loguru import logger

try:
    import cQIGen as qinfer
except ImportError:
    logger.error('cQIGen not installed.')
    raise

def mem_model(N, M, T, mu, tu, bits, l1, p, gs):
    m = GEKKO() # create GEKKO model
    #cinfergen if bits==3:
        # tu = tu*3
    B = m.Const(value=bits)
    TP = m.Const(value=T//p)
    k = m.Var(1,integer=True,lb=1)
    z = m.Var(1,integer=True,lb=1)
    w = m.Var(1,integer=True,lb=1)
    y = m.Var(1,integer=True,lb=1)
    v = m.Var(1,integer=True,lb=1)
    mb = m.Var(mu,integer=True,lb=1)
    if gs != -1:
        gg = m.Var(1,integer=True,lb=1)
    tb = m.Var(tu,integer=True,lb=1,ub=int(T/p))
    L = m.Var(integer=True,lb=0,ub=l1)
    m.Equation(L == 32 * mb * N + B * mb * tb + 32 * tb * N)
    m.Equation(mb * k == M)
    if gs != -1:
        m.Equation(gs * gg == mb)
    # m.Equation(tb * z == T)
    m.Equation(tb * z == TP)
    m.Equation(mu * w == mb)
    m.Equation(tu * y == tb)
    # m.Equation(tb * v == tt)
    m.Maximize(L)
    m.options.SOLVER = 1
    m.solver_options = ['minlp_maximum_iterations 1000', \
                # minlp iterations with integer solution
                'minlp_max_iter_with_int_sol 10', \
                # treat minlp as nlp
                'minlp_as_nlp 0', \
                # nlp sub-problem max iterations
                'nlp_maximum_iterations 100', \
                # 1 = depth first, 2 = breadth first
                'minlp_branch_method 2', \
                # maximum deviation from whole number
                'minlp_integer_tol 0.00', \
                # covergence tolerance
                'minlp_gap_tol 0.01']
    try:
        m.solve(disp=False)
    except:
        try:
            m.solver_options = ['minlp_maximum_iterations 1000', \
                            # minlp iterations with integer solution
                            'minlp_max_iter_with_int_sol 10', \
                            # treat minlp as nlp
                            'minlp_as_nlp 0', \
                            # nlp sub-problem max iterations
                            'nlp_maximum_iterations 100', \
                            # 1 = depth first, 2 = breadth first
                            'minlp_branch_method 1', \
                            # maximum deviation from whole number
                            'minlp_integer_tol 0.00', \
                            # covergence tolerance
                            'minlp_gap_tol 0.01']
            m.solve(disp=False)
        except:
            # mytb = T//p
            mytb = tu
            if gs != -1:
                mymb = gs
                while 32 * (mymb + gs) * N + bits * (mymb + gs) * mytb + 32 * mytb * N < l1:
                    mymb += gs
                while M % mymb != 0:
                    mymb -= gs
                return (int(mymb), int(mytb))
            else:
                mymb = mu
                while 32 * (mymb + mu) * N + bits * (mymb + mu) * mytb + 32 * mytb * N < l1:
                    mymb += mu
                while M % mymb != 0:
                    mymb -= mu
                return (int(mymb), int(mytb))

    return (int(mb.value[0]), int(tb.value[0]))

params = {}

def compute_reductions(x, gs=-1, cpp=True):
    if cpp:
        if len(x.shape) != 1:
            rows, cols = x.shape
        else:
            rows = 1
            cols = x.shape[0]
        if gs == -1:
            out = torch.zeros(rows).float().contiguous()
            mygs = cols
        else:
            out = torch.zeros(rows, cols // gs).float().contiguous()
            mygs = gs
        
        qinfer.compute_reduction_cpp(x, out, rows, cols, mygs)
        return out
    if gs == -1: 
        if len(x.shape) != 1:
            return torch.sum(x,1)
        else:
            return torch.sum(x)
    else:
        if len(x.shape) != 1:
            rows, cols = x.shape
            out = torch.zeros(rows, cols // gs).float().contiguous()
            for i in range(cols // gs):
                out[:,i] = torch.sum(x[:,i*gs:(i+1)*gs],1)
            return out
        else:
            cols = x.shape[0]
            out = torch.zeros(cols // gs).float().contiguous()
            for i in range(cols // gs):
                out[i] = torch.sum(x[i*gs:(i+1)*gs])
            return out

def process_zeros_scales(zeros, scales, bits, M):
    if zeros.dtype != torch.float32:
        new_zeros = torch.zeros_like(scales).float().contiguous()
        if bits == 4:
            qinfer.unpack_zeros4(zeros, new_zeros, new_zeros.shape[0], new_zeros.shape[1])
        elif bits == 2:
            qinfer.unpack_zeros2(zeros, new_zeros, new_zeros.shape[0], new_zeros.shape[1])
        elif bits == 3:
            logger.info("Unpacking zeros for 3 bits")
        new_scales = scales.contiguous()
    else:
        if scales.shape[1] != M:
            new_scales = scales.transpose(0,1).contiguous()
        else:
            new_scales = scales.contiguous()
        if zeros.shape[1] != M:
            new_zeros = zeros.transpose(0,1).contiguous()
        else:
            new_zeros = zeros.contiguous()

    return new_zeros, new_scales