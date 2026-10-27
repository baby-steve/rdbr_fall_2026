import numpy as np

def ground_contact(p, pd):
    # Stiffness of ground plan
    k = 50_000.0
    # Damping of the ground plane
    d = 100.0
    # Friction of the ground plane
    mu = 0.1

    # Normal force calculation
    fn = []

    z = p[:, 2]
    zd = pd[:, 2]
    zr = np.sqrt(np.maximum(0, -z))

    fn = zr * ((-k) * z - d * zd)

    if mu == 0:
        return [0, 0, np.maximum(0, fn)] # easy case with no friction

    # Step 1: set output variable to correct no-contact values
    # n_d: Number of dimensions (should be 3)
    # n_p: Number of points (for quadruped this is 8)
    (n_p, n_d) = np.shape(p)

    # Tangential deformation of the ground plane
    u = np.zeros((n_p, n_d - 1))

    f = np.zeros((n_p, n_d))
    
    ud = (-k/d) * u
    fcone = np.tile(2, (n_p, 1)) # 2 is just some arbitary value > 1
    
    in_ctact = fn > 0
    
    # Step 2: correct all those values that are wrong because their respective points
    # are in contact. 
    # Note: variables with names ending in c have one column for each poin that is
    # in contact.
    if any(in_ctact):
        # Step 2a: calculate correct values for sticking
        fnc = fn[in_ctact]
        udc = pd[in_ctact, :-1]
        
        zrc = np.tile(zr[in_ctact], (1, 1)).T
        kc = k * zrc
        dc = d * zrc
        
        fkc = kc * u[in_ctact, :]
        
        ftc = -fkc - dc * udc # correct tangent force for sticking

        # now test for slipping
        fslip = mu * fnc
        ftcmag = np.sqrt(np.sum(ftc**2, 1))
        fconec = ftcmag / fslip
        slipping = fconec > 1

        # step 2b: adjust values for those points that are found to be slipping
        if any(slipping):
            attenuator = np.tile(fconec[slipping], (n_d - 1, 1)).T
            fts = ftc[slipping, :] / attenuator
            ftc[slipping,:] = fts
            udc[slipping,:] = -(fts + fkc[slipping,:]) / dc[slipping,:]

        # step 2c: apply the corrections to the output variables
        f[in_ctact, :-1] = ftc
        f[in_ctact, 2] = fnc
        ud[in_ctact, :] = udc

        (n,) = fconec.shape

        fcone[in_ctact, :] = np.minimum(2, fconec).reshape((n,1))

    return f
