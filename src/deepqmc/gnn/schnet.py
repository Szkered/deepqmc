import haiku as hk
import jax.numpy as jnp
from jax import ops

from ..hkext import MLP
from .edge_features import PauliNetEdgeFeatures
from .gnn import GraphNeuralNetwork, MessagePassingLayer
from .graph import GraphNodes, difference_callback


class SchNetLayer(MessagePassingLayer):
    r"""
    The message passing layer of :class:`SchNet`.

    Derived from :class:`~deepqmc.gnn.gnn.MessagePassingLayer`.

    Args:
        ilayer (int): the index of the current layer in the list of all layers
        shared (dict): attribute names and values which are shared between the
            layers and the :class:`SchNet` instance.
        shared_h (bool): optional, whether to use a shared :data:`h` subnetwork.
        shared_g (bool): optional, whether to use a shared :data:`g` subnetwork.
        n_layers_w (int): optional, the number of layers in the :data:`w`
            subnetwork.
        n_layers_h (int): optional, the number of layers in the :data:`h`
            subnetwork.
        n_layers_g (int): optional, the number of layers in the :data:`g`
            subnetwork.
        subnet_kwargs (dict): optional, extra arguments passed to the
            :class:`~deepqmc.hkext.MLP` constructor of the subnetworks.
        subnet_kwargs_by_lbl (dict): optional, extra arguments passed to the
            :class:`~deepqmc.hkext.MLP` constructor of the subnetworks. Arguments
            can be specified independently for each subnet
            (:data:`w`, :data:`h` or :data:`g`).
    """

    def __init__(
        self,
        *,
        shared_h=False,
        shared_g=False,
        n_layers_w=3,
        n_layers_h=3,
        n_layers_g=2,
        subnet_kwargs=None,
        subnet_kwargs_by_lbl=None,
        residual=True,
        sum_z=False,
        deep_w=False,
        **layer_attrs,
    ):
        super().__init__(**layer_attrs)
        self.shared_h = shared_h
        self.shared_g = shared_g
        assert shared_g or not sum_z
        self.sum_z = sum_z
        self.deep_w = deep_w
        default_n_layers = {'w': n_layers_w, 'h': n_layers_h, 'g': n_layers_g}

        subnet_kwargs = subnet_kwargs or {}
        subnet_kwargs.setdefault('last_linear', True)
        subnet_kwargs.setdefault('activation', jnp.tanh)

        subnet_kwargs_by_lbl = subnet_kwargs_by_lbl or {}
        for lbl in self.subnet_labels:
            subnet_kwargs_by_lbl.setdefault(lbl, {})
            for k, v in subnet_kwargs.items():
                subnet_kwargs_by_lbl[lbl].setdefault(k, v)
            subnet_kwargs_by_lbl[lbl].setdefault('bias', lbl != 'w')
            subnet_kwargs_by_lbl[lbl].setdefault(
                'hidden_layers', ('log', default_n_layers[lbl])
            )

        self.w = {
            typ: MLP(
                (
                    self.edge_feat_dim[typ]
                    if not deep_w or self.first_layer
                    else self.embedding_dim
                ),
                self.embedding_dim,
                name=f'w_{typ}',
                **subnet_kwargs_by_lbl['w'],
            )
            for typ in self.edge_types
        }

        def h_factory(typ=None):
            name = 'h' if shared_h else f'h_{typ}'
            return MLP(
                self.embedding_dim,
                self.embedding_dim,
                name=name,
                **subnet_kwargs_by_lbl['h'],
            )

        self.h = (
            h_factory()
            if shared_h
            else {typ: h_factory(typ) for typ in self.edge_types}
        )
        self.g = (
            MLP(
                self.embedding_dim,
                self.embedding_dim,
                name='g',
                **subnet_kwargs_by_lbl['g'],
            )
            if shared_g
            else {
                typ: MLP(
                    self.embedding_dim
                    + (2 * self.embedding_dim if shared_g and not sum_z else 0),
                    self.embedding_dim,
                    name=f'g_{typ}',
                    **subnet_kwargs_by_lbl['g'],
                )
                for typ in self.edge_types
            }
        )
        self.residual = residual
        self.f = hk.Linear(1, with_bias=False, w_init=jnp.ones, name='f')

    @classmethod
    @property
    def subnet_labels(cls):
        return ('w', 'h', 'g')

    def get_update_edges_fn(self):
        def update_edges(edges):
            updated_edges = {
                typ: edge._replace(features=self.w[typ](edge.features))
                for typ, edge in edges.items()
            }
            return updated_edges if self.deep_w else edges

        return update_edges

    def get_aggregate_edges_for_nodes_fn(self):
        def aggregate_edges_for_nodes(nodes, edges):
            if self.deep_w:
                we = {typ: edge.features for typ, edge in edges.items()}
            else:
                we = {typ: self.w[typ](edges[typ].features) for typ in self.edge_types}
            if self.shared_h:
                hx = nodes.electrons if self.first_layer else self.h(nodes.electrons)
                hx = {typ: hx[edges[typ].senders] for typ in self.edge_types[:2]}
            else:
                hx = {
                    typ: (
                        nodes.electrons
                        if self.first_layer
                        else self.h[typ](nodes.electrons)
                    )[edges[typ].senders]
                    for typ in self.edge_types[:2]
                }
            wh = {
                typ: we[typ] * hx[typ] for typ in self.mapping.with_sender('electrons')
            }
            wh['ne'] = we['ne'] * nodes.nuclei[edges['ne'].senders]
            z = {
                typ: ops.segment_sum(
                    data=wh[typ],
                    segment_ids=edges[typ].receivers,
                    num_segments=self.mapping.receiver_data_of(typ, 'n_nodes'),
                )
                for typ in self.edge_types
            }
            return z

        return aggregate_edges_for_nodes

    def get_update_nodes_fn(self):
        def update_nodes(nodes, z):
            if self.shared_g:
                if self.sum_z:
                    zs = jnp.stack(list(z.values()), axis=-1)
                    fz = self.f(zs).squeeze(axis=-1)
                    updated = self.g(fz) + (nodes.electrons if self.residual else 0)
                else:
                    zs = jnp.concatenate(
                        [nodes.electrons, z['same'] + z['anti'], z['ne']],
                        axis=-1,
                    )
                    updated = self.g(zs)
            else:
                gs = jnp.stack(
                    [self.g[typ](z[typ]) for typ in self.edge_types], axis=-1
                )
                if self.residual:
                    gs = jnp.concatenate([nodes.electrons[..., None], gs], axis=-1)
                updated = self.f(gs).squeeze(axis=-1)
            nodes = GraphNodes(nodes.nuclei, updated)

            return nodes

        return update_nodes


class SchNet(GraphNeuralNetwork):
    r"""
    The SchNet architecture [SchuettJCP18]_ adapted for graphs of nuclei and electrons.

    Derived from :class:`~deepqmc.gnn.gnn.GraphNeuralNetwork`.

    Args:
        mol (~deepqmc.Molecule): the molecule on which the graph is defined.
        embedding_dim (int): the length of the electron embedding vectors.
        n_interactions (int): number of message passing interactions.
        edge_feat_kwargs (dict): extra arguments passed to
            :class:`~deepqmc.gnn.edge_features.EdgeFeatures`.
        edge_feat_kwargs_by_typ (dict): extra arguments passed to
            :class:`~deepqmc.gnn.edge_features.EdgeFeatures`, specified
            indepenedently for different edge types.
        gnn_kwargs (dict): extra arguments passed to the
            :class:`~deepqmc.gnn.gnn.GraphNeuralNetwork` base class.
    """

    def __init__(
        self,
        mol,
        embedding_dim,
        *,
        distance_basis_radius=30.0,
        n_interactions=3,
        edge_feat_kwargs=None,
        edge_feat_kwargs_by_typ=None,
        **gnn_kwargs,
    ):
        n_nuc, n_up, n_down = mol.n_particles
        edge_feat_kwargs = edge_feat_kwargs or {}
        edge_feat_kwargs.setdefault('feature_dim', 32)
        edge_feat_kwargs.setdefault('cutoff', distance_basis_radius)
        edge_feat_kwargs.setdefault('powers', [1])
        edge_feat_kwargs.setdefault('difference', True)
        edge_feat_kwargs_by_typ = edge_feat_kwargs_by_typ or {}
        for typ in self.edge_types:
            edge_feat_kwargs_by_typ.setdefault(typ, {})
            for k, v in edge_feat_kwargs.items():
                edge_feat_kwargs_by_typ[typ].setdefault(k, v)
        super().__init__(
            mol,
            embedding_dim,
            n_interactions,
            **gnn_kwargs,
            layer_attrs={
                'edge_feat_dim': {
                    typ: edge_feat_kwargs_by_typ[typ]['feature_dim']
                    for typ in self.edge_types
                },
            },
        )
        self.edge_features = {
            typ: PauliNetEdgeFeatures(**kwargs)
            for typ, kwargs in edge_feat_kwargs_by_typ.items()
        }

    def node_factory(self):
        n_elec_types = self.node_data['n_node_types']['electrons']
        n_nuc_types = self.node_data['n_node_types']['nuclei']
        X = hk.Embed(n_elec_types, self.embedding_dim, name='ElectronicEmbedding')
        Y = hk.Embed(n_nuc_types, self.embedding_dim, name='NuclearEmbedding')
        return GraphNodes(
            Y(self.node_data['node_types']['nuclei'] - n_elec_types),
            X(self.node_data['node_types']['electrons']),
        )

    def init_state(self, shape, dtype):
        zeros = jnp.zeros(shape, dtype)
        return {'anti': (zeros, zeros), 'ne': zeros, 'same': (zeros, zeros)}

    @classmethod
    @property
    def edge_types(cls):
        return ('same', 'anti', 'ne')

    def edge_feature_callback(self, typ, *feature_callback_args):
        r = difference_callback(*feature_callback_args)
        return self.edge_features[typ](r)

    @classmethod
    @property
    def layer_factory(cls):
        return SchNetLayer
