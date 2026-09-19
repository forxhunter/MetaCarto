import cobra

def run_fba(model):
    """
    Runs FBA on the model to maximize the objective function.
    Returns the solution object.
    """
    if not model:
        return None
        
    try:
        solution = model.optimize()
        print(f"FBA Status: {solution.status}")
        print(f"Objective Value: {solution.objective_value}")
        return solution
    except Exception as e:
        print(f"Error running FBA: {e}")
        return None

def get_flux_weights(model, solution):
    """
    Returns a dictionary of reaction IDs and their absolute flux values.
    Normalized to range [0, 1] for weighting.
    """
    if solution.status != 'optimal':
        return {}
        
    fluxes = solution.fluxes.abs()
    max_flux = fluxes.max()
    
    if max_flux == 0:
        return fluxes.to_dict()
        
    normalized_fluxes = (fluxes / max_flux).to_dict()
    return normalized_fluxes
