Install via pip install .
Import as complex_inversion_tools

The inversion uses the complex coordinate representation of complex vectors. For the complex vector $\tilde{\mathbf{x}} \in \mathbb{C}^M$ we define $\stackrel{c}{\mathbf{x}}\,\stackrel{\Delta}{\mathbf{=}}\left(\tilde{\mathbf{x}}, \tilde{\mathbf{x}}^*\right)^T \in \mathbb{C}^{2M}$ to be its representation in conjugate coordinates, with $^*$ denoting the complex conjugation and $^T$ denoting the transpose.
The representation in conjugate coordinates is related to the representation of $\tilde{\mathbf{x}}$ in terms of its real and imaginary part $\stackrel{r}{\mathbf{x}}\,\stackrel{\Delta}{\mathbf{=}} (\mathbf{x}', \mathbf{x}'')^T \in \mathbb{R}^{2M}$ by the linear transformations

\begin{align}
    \stackrel{c}{\mathbf{x}}\,=\,\underbrace{\left[\begin{array}{cr}
        \mathbf{I}     & i\mathbf{I} \\
        \mathbf{I}     &  -i\mathbf{I}
    \end{array}\right]}_{\tilde{\mathbf{S}}}\stackrel{r}{\mathbf{x}}
\end{align}

\noindent and 

\begin{align}
    \stackrel{r}{\mathbf{x}}\,=\,\tilde{\mathbf{S}}^{-1}\stackrel{c}{\mathbf{x}}\,=\,\frac{1}{2}\tilde{\mathbf{S}}^H\stackrel{c}{\mathbf{x}},
\end{align}

\noindent with $^H$ denoting the conjugate transpose, $\mathbf{I}$ denoting the identity matrix and $i^2 = -1$. Input vectors (for example for model and data) have to be formulated accordingly.
