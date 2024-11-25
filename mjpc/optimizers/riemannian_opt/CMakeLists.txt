set(RIE_OPT_HEADERS
        Convex/Concepts.h
        Convex/ADMM.h
        Convex/ProximalGradient.h
        LinearAlgebra/Concepts.h
        LinearAlgebra/LOBPCG.h
        LinearAlgebra/IterativeSolvers.h
        Riemannian/GradientDescent.h
        Riemannian/Concepts.h
        Riemannian/TNT.h
        Riemannian/TNLS.h
        Base/Concepts.h
        Util/Stopwatch.h
)

target_sources(
        libmjpc PRIVATE
        ${RIE_OPT_HEADERS}
)

