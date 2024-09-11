import numpy as np


def find_first_peak(wf, th_Pmax):
    """
    Find the first peak above th_Pmax
    """

    for i in range(1, len(wf)):
        # Check on Pmax
        if wf[i] <= th_Pmax:
            continue
        # Search for local maximum
        if wf[i] > wf[i-1] and wf[i] > wf[i+1]:
            return i, wf[i]
    return []


def tfmra_retracker(P, th_rtck, th_P, gates_Pn):
    """
    Return the retracking gate and the index of the selected peak

    Input:
        - P             waveform power
        - th_rtck       Ppeak percentage to retrack
        - th_P          Pmax percentage to avoid retracking noise in front of
                        the leading edge
        - gates_Pn      number of gates used to compute thermal noise
    Output:
        - gate_rtck     retracking gate
        - ipeak         index of the retracked peak
    """

    ### Pre-processing
    P       = list(P)
    gates   = range(len(P))
    AmpPmax    = max(P)                  # added by Henriette

    ### Retracking
    # Find first peak
    ipeak, ipeakAmp   = find_first_peak(P, max(P)*th_P)

    # Check if no peak is found
    if ipeak == []:
        print("No peak found!")
        gate_rtck = 'NaN'
        ipeak = 'NaN'
        return float(gate_rtck), float(ipeak)  

    # If first peak is in the first or last gates_Pn gates, WF is discarded
    if ipeak < gates_Pn or ipeak > len(P) - gates_Pn:
        print("First peak coincides with thermal noise!")
        gate_rtck = 'NaN'
        ipeak = 'NaN'
        return float(gate_rtck), float(ipeak)  

    # Saving first peak power value
    Ppeak   = P[ipeak]

    # Compute thermal noise
    Pn      = np.mean(P[:gates_Pn])

    # Defining the power threshold
    th_tfmra = Pn + th_rtck * (Ppeak-Pn)

    # Find the first gate exceeding the power threshold:
    # searched after gates_Pn because the retracked point cannot be at the
    # gates used to compute thermal noise
    Ptemp   = next(x for x in P[gates_Pn:] if x > th_tfmra)
    itemp   = P.index(Ptemp)

    # Calculate the exact retracked gate
    gate_rtck = gates[itemp]-1 + (th_tfmra-P[itemp-1]) / (P[itemp]-P[itemp-1])


    return gate_rtck, ipeak, AmpPmax, ipeakAmp
