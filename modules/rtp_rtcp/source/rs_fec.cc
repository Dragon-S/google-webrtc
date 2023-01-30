

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <stdint.h>
// #include <iostream>
//using namespace std;
// #include "webrtc/typedefs.h"
#include "rtc_base/logging.h"
#include "rs_fec.h"

#if __cplusplus > 199711L
#define register      // Deprecated in C++11.
#endif  // #if __cplusplus > 199711L


#define GF_BITS  8
#define	GF_SIZE ((1 << GF_BITS) - 1)


typedef unsigned char gf;

static gf gf_exp[2*GF_SIZE];	/* index->poly form conversion table	*/
static int gf_log[GF_SIZE + 1];	/* Poly->index form conversion table	*/
static gf inverse[GF_SIZE+1];	/* inverse of field elem.		*/
static gf gf_mul_table[GF_SIZE + 1][GF_SIZE + 1];

#define SWAP(a,b,t) {t tmp; tmp=a; a=b; b=tmp;}

//static gf gf_mul_table[GF_SIZE + 1][GF_SIZE + 1];

#define gf_mul(x,y) gf_mul_table[x][y]

#define USE_GF_MULC register gf * __gf_mulc_
#define GF_MULC0(c) __gf_mulc_ = gf_mul_table[c]
#define GF_ADDMULC(dst, x) dst ^= __gf_mulc_[x]

#define addmul(dst, src, c, sz) \
    if (c != 0) addmul1(dst, src, c, sz)


static inline gf modnn(int x)
{
    while (x >= GF_SIZE) {
	x -= GF_SIZE;
	x = (x >> GF_BITS) + (x & GF_SIZE);
    }
    return x;
}

#define UNROLL 16 /* 1, 4, 8, 16 */
static void addmul1(gf *dst1, gf *src1, gf c, int sz)
{
    USE_GF_MULC ;
    register gf *dst = dst1, *src = src1 ;
    gf *lim = &dst[sz - UNROLL + 1] ;

    GF_MULC0(c) ;

#if (UNROLL > 1) /* unrolling by 8/16 is quite effective on the pentium */
    for (; dst < lim ; dst += UNROLL, src += UNROLL ) {
	GF_ADDMULC( dst[0] , src[0] );
	GF_ADDMULC( dst[1] , src[1] );
	GF_ADDMULC( dst[2] , src[2] );
	GF_ADDMULC( dst[3] , src[3] );
#if (UNROLL > 4)
	GF_ADDMULC( dst[4] , src[4] );
	GF_ADDMULC( dst[5] , src[5] );
	GF_ADDMULC( dst[6] , src[6] );
	GF_ADDMULC( dst[7] , src[7] );
#endif
#if (UNROLL > 8)
	GF_ADDMULC( dst[8] , src[8] );
	GF_ADDMULC( dst[9] , src[9] );
	GF_ADDMULC( dst[10] , src[10] );
	GF_ADDMULC( dst[11] , src[11] );
	GF_ADDMULC( dst[12] , src[12] );
	GF_ADDMULC( dst[13] , src[13] );
	GF_ADDMULC( dst[14] , src[14] );
	GF_ADDMULC( dst[15] , src[15] );
#endif
    }
#endif
    lim += UNROLL - 1 ;
    for (; dst < lim; dst++, src++ )		/* final components */
	GF_ADDMULC( *dst , *src );
}


/*
 * initialize the data structures used for computations in GF.
 */
static void generate_gf(void)
{
    int i;
    gf mask;
    char Pp[] =  "101110001";

    mask = 1;	/* x ** 0 = 1 */
    gf_exp[GF_BITS] = 0; /* will be updated at the end of the 1st loop */
    /*
     * first, generate the (polynomial representation of) powers of \alpha,
     * which are stored in gf_exp[i] = \alpha ** i .
     * At the same time build gf_log[gf_exp[i]] = i .
     * The first GF_BITS powers are simply bits shifted to the left.
     */
    for (i = 0; i < GF_BITS; i++, mask <<= 1 ) {
	gf_exp[i] = mask;
	gf_log[gf_exp[i]] = i;
	/*
	 * If Pp[i] == 1 then \alpha ** i occurs in poly-repr
	 * gf_exp[GF_BITS] = \alpha ** GF_BITS
	 */
	if ( Pp[i] == '1' )
	    gf_exp[GF_BITS] ^= mask;
    }
    /*
     * now gf_exp[GF_BITS] = \alpha ** GF_BITS is complete, so can als
     * compute its inverse.
     */
    gf_log[gf_exp[GF_BITS]] = GF_BITS;
    /*
     * Poly-repr of \alpha ** (i+1) is given by poly-repr of
     * \alpha ** i shifted left one-bit and accounting for any
     * \alpha ** GF_BITS term that may occur when poly-repr of
     * \alpha ** i is shifted.
     */
    mask = 1 << (GF_BITS - 1 ) ;
    for (i = GF_BITS + 1; i < GF_SIZE; i++) {
	if (gf_exp[i - 1] >= mask)
	    gf_exp[i] = gf_exp[GF_BITS] ^ ((gf_exp[i - 1] ^ mask) << 1);
	else
	    gf_exp[i] = gf_exp[i - 1] << 1;
	gf_log[gf_exp[i]] = i;
    }
    /*
     * log(0) is not defined, so use a special value
     */
    gf_log[0] =	GF_SIZE ;
    /* set the extended gf_exp values for fast multiply */
    for (i = 0 ; i < GF_SIZE ; i++)
	gf_exp[i + GF_SIZE] = gf_exp[i] ;

    /*
     * again special cases. 0 has no inverse. This used to
     * be initialized to GF_SIZE, but it should make no difference
     * since noone is supposed to read from here.
     */
    inverse[0] = 0 ;
    inverse[1] = 1;
    for (i=2; i<=GF_SIZE; i++)
	inverse[i] = gf_exp[GF_SIZE-gf_log[i]];
}

static void init_mul_table()
{
    int i, j;
    for (i=0; i< GF_SIZE+1; i++)
	for (j=0; j< GF_SIZE+1; j++)
	    gf_mul_table[i][j] = gf_exp[modnn(gf_log[i] + gf_log[j]) ] ;

    for (j=0; j< GF_SIZE+1; j++)
	    gf_mul_table[0][j] = gf_mul_table[j][0] = 0;
}

static int fec_initialized = 0 ;
static void init_fec()
{
	
	if(fec_initialized == 0)
	{
		fec_initialized = 1;

	    generate_gf();
	    
	    init_mul_table();	    
	}

}

static int invert_vdm(gf *src, int k)
{
    int i, j, row, col ;
    gf *b, *c, *p;
    gf t, xx ;

    if (k == 1) 	/* degenerate case, matrix must be p^0 = 1 */
	return 0 ;
    /*
     * c holds the coefficient of P(x) = Prod (x - p_i), i=0..k-1
     * b holds the coefficient for the matrix inversion
     */
    c = new gf[k];
    b = new gf[k];

    p = new gf[k];
   
    for ( j=1, i = 0 ; i < k ; i++, j+=k ) 
    {
		c[i] = 0 ;
		p[i] = src[j] ;    /* p[i] */
    }
    /*
     * construct coeffs. recursively. We know c[k] = 1 (implicit)
     * and start P_0 = x - p_0, then at each stage multiply by
     * x - p_i generating P_i = x P_{i-1} - p_i P_{i-1}
     * After k steps we are done.
     */
    c[k-1] = p[0] ;	/* really -p(0), but x = -x in GF(2^m) */
    for (i = 1 ; i < k ; i++ ) 
    {
		gf p_i = p[i] ; /* see above comment */
		for (j = k-1  - ( i - 1 ) ; j < k-1 ; j++ )
		    c[j] ^= gf_mul( p_i, c[j+1] ) ;
		
		c[k-1] ^= p_i ;
    }

    for (row = 0 ; row < k ; row++ ) {
	/*
	 * synthetic division etc.
	 */
	xx = p[row] ;
	t = 1 ;
	b[k-1] = 1 ; /* this is in fact c[k] */
	for (i = k-2 ; i >= 0 ; i-- ) {
	    b[i] = c[i+1] ^ gf_mul(xx, b[i+1]) ;
	    t = gf_mul(xx, t) ^ b[i] ;
	}
	for (col = 0 ; col < k ; col++ )
	    src[col*k + row] = gf_mul(inverse[t], b[col] );
    }

    delete [] c;
    delete [] b;
    delete [] p;
    return 0 ;
}

static void matmul(gf *a, gf *b, gf *c, int n, int k, int m)
{
    int row, col, i ;

    for (row = 0; row < n ; row++) {
	for (col = 0; col < m ; col++) {
	    gf *pa = &a[ row * k ];
	    gf *pb = &b[ col ];
	    gf acc = 0 ;
	    for (i = 0; i < k ; i++, pa++, pb += m )
		acc ^= gf_mul( *pa, *pb ) ;
	    c[ row * m + col ] = acc ;
	}
    }
}

static gf *creat_encode_mat(int n, int k)
{
    int row, col ;
    gf *p, *tmp_m, *em ;

    em = new gf[n*k];
    
    tmp_m = new gf[n*k];
    /*
     * fill the matrix with powers of field elements, starting from 0.
     * The first row is special, cannot be computed with exp. table.
     */
    tmp_m[0] = 1 ;
    for (col = 1; col < k ; col++)
	tmp_m[col] = 0 ;
    for (p = tmp_m + k, row = 0; row < n-1 ; row++, p += k) {
	for ( col = 0 ; col < k ; col ++ )
	    p[col] = gf_exp[modnn(row*col)];
    }

    /*
     * quick code to build systematic matrix: invert the top
     * k*k vandermonde matrix, multiply right the bottom n-k rows
     * by the inverse, and construct the identity matrix at the top.
     */

    invert_vdm(tmp_m, k); /* much faster than invert_mat */
    matmul(tmp_m + k*k, tmp_m, em + k*k, n - k, k, k);
    /*
     * the upper matrix is I so do not bother with a slow multiply
     */
    memset(em, 0,k*k*sizeof(gf) );
    for (p = em, col = 0 ; col < k ; col++, p += k+1 )
    {
		*p = 1 ;
    }
    delete [] tmp_m;

    return em;
}




/*
 * fec_encode accepts as input pointers to n data packets of size sz,
 * and produces as output a packet pointed to by fec, computed
 * with index "index".
 */
static void fec_encode(gf *code, int n, int k, gf *src[], gf *fec, int index, int sz)
{
    int i ;
    gf *p ;



    if (index < k)
    {
        memcpy(fec, src[index], sz*sizeof(gf) ) ;
    }
    else if (index < n) 
    {
			p = &(code[index*k]);
			memset(fec, 0, sz*sizeof(gf));
			for (i = 0; i < k ; i++)
			    addmul(fec, src[i], p[i], sz ) ;
    } else
	fprintf(stderr, "Invalid index %d (max %d)\n",
	    index, n - 1 );
}


static int invert_mat(gf *src, int k)
{
    gf c, *p ;
    int irow, icol, row, col, i, ix ;

    int error = 1 ;


    int *indxc = new int[k];
    int *indxr = new int[k];
    int *ipiv = new int[k];
    gf *id_row = new gf[k];
    

    memset(id_row, 0, k*sizeof(gf));
    
    /*
     * ipiv marks elements already used as pivots.
     */
    for (i = 0; i < k ; i++)
	ipiv[i] = 0 ;

    for (col = 0; col < k ; col++) {
	gf *pivot_row ;
	/*
	 * Zeroing column 'col', look for a non-zero element.
	 * First try on the diagonal, if it fails, look elsewhere.
	 */
	irow = icol = -1 ;
	if (ipiv[col] != 1 && src[col*k + col] != 0) 
	{
	    irow = col ;
	    icol = col ;
	    
	}
	else
	{
		for (row = 0 ; row < k ; row++) 
		{
		    if (ipiv[row] != 1) 
		    {
				for (ix = 0 ; ix < k ; ix++) 
				{
				    
				    if (ipiv[ix] == 0) 
				    {
						if (src[row*k + ix] != 0) 
						{
						    irow = row ;
						    icol = ix ;
						    break ;
						}
				    } 
				    else if (ipiv[ix] > 1) 
				    {
						fprintf(stderr, "singular matrix\n");
						goto fail ; 
				    }
				}
		    }
		}
		if (icol == -1) {
		    fprintf(stderr, "XXX pivot not found!\n");
		    goto fail ;
		}
	}

	++(ipiv[icol]) ;
	/*
	 * swap rows irow and icol, so afterwards the diagonal
	 * element will be correct. Rarely done, not worth
	 * optimizing.
	 */
	if (irow != icol) {
	    for (ix = 0 ; ix < k ; ix++ ) {
		SWAP( src[irow*k + ix], src[icol*k + ix], gf) ;
	    }
	}
	indxr[col] = irow ;
	indxc[col] = icol ;
	pivot_row = &src[icol*k] ;
	c = pivot_row[icol] ;
	if (c == 0) {
	    fprintf(stderr, "singular matrix 2\n");
	    goto fail ;
	}
	if (c != 1 ) { /* otherwhise this is a NOP */
	    /*
	     * this is done often , but optimizing is not so
	     * fruitful, at least in the obvious ways (unrolling)
	     */
	    
	    c = inverse[ c ] ;
	    pivot_row[icol] = 1 ;
	    for (ix = 0 ; ix < k ; ix++ )
		pivot_row[ix] = gf_mul(c, pivot_row[ix] );
	}
	/*
	 * from all rows, remove multiples of the selected row
	 * to zero the relevant entry (in fact, the entry is not zero
	 * because we know it must be zero).
	 * (Here, if we know that the pivot_row is the identity,
	 * we can optimize the addmul).
	 */
	id_row[icol] = 1;
	if (memcmp(pivot_row, id_row, k*sizeof(gf)) != 0) {
	    for (p = src, ix = 0 ; ix < k ; ix++, p += k ) {
		if (ix != icol) {
		    c = p[icol] ;
		    p[icol] = 0 ;
		    addmul(p, pivot_row, c, k );
		}
	    }
	}
	id_row[icol] = 0;
    } /* done all columns */
    for (col = k-1 ; col >= 0 ; col-- ) {
	if (indxr[col] <0 || indxr[col] >= k)
	    fprintf(stderr, "AARGH, indxr[col] %d\n", indxr[col]);
	else if (indxc[col] <0 || indxc[col] >= k)
	    fprintf(stderr, "AARGH, indxc[col] %d\n", indxc[col]);
	else
	if (indxr[col] != indxc[col] ) {
	    for (row = 0 ; row < k ; row++ ) {
		SWAP( src[row*k + indxr[col]], src[row*k + indxc[col]], gf) ;
	    }
	}
    }
    error = 0 ;


fail:
    delete [] indxc;
    delete [] indxr;
    delete [] ipiv;
    delete [] id_row;
    
    return error ;
}


static int shuffle(gf *pkt[], int index[], int k)
{
    int i;

    for ( i = 0 ; i < k ; ) 
    {
		if (index[i] >= k || index[i] == i)
		{
		    i++ ;
		}
		else 
		{
		    /*
		     * put pkt in the right position (first check for conflicts).
		     */
		    int c = index[i] ;

		    if (index[c] == c) 
		    {
		    	return 1;
		    }
		    SWAP(index[i], index[c], int) ;
		    SWAP(pkt[i], pkt[c], gf *) ;
		}
    }
    
    for ( i = 0 ; i < k ; i++ ) 
    {
		if (index[i] < k && index[i] != i) 
		{
		    fprintf(stderr, "shuffle: after\n");
		    for (i=0; i<k ; i++) fprintf(stderr, "%3d ", index[i]);
		    fprintf(stderr, "\n");
		    return 1 ;
		}
    }
    
    return 0 ;
}

static gf *build_decode_matrix(gf *code, int n, int k, int index[])
{
    int i;
    gf *p, *matrix;
    matrix = new gf[k*k];

    for (i = 0, p = matrix ; i < k ; i++, p += k )
    {
		if(index[i] < k) 
		{
		    memset(p,0, k*sizeof(gf));
		    p[i] = 1 ;
		} 
		else if(index[i] < n )
		{			
			{
			    memcpy( p, &(code[index[i]*k]),  k*sizeof(gf) ); 
			}
		}
		else 
		{
		    delete [] matrix ;
		    return NULL ;
		}
    }

    if (invert_mat(matrix, k) != 0) 
    {
		delete [] matrix ;
		matrix = NULL ;
    }

    return matrix ;
}


static int fec_decode(gf *code, int n, int k, gf *pkt[], int index[], int sz)
{
    gf *m_dec ; 
    gf **new_pkt ;
    int row, col ;

    // cout <<"fec_decode n: "  << n <<", k: "  << k <<endl;

    if (code == NULL)
    {
		return 1 ; /* error */
    }

    for (col = 0 ; col < k ; col++ )
    {
    	if(pkt[col]==NULL)
    	{
    		
    		return 1;
    	}
	}

    if (shuffle(pkt, index, k))	
    {
		return 1 ;
    }
    m_dec = build_decode_matrix(code,n,k, index);

    if (m_dec == NULL)
    {
		return 1 ; /* error */
    }
    /*
     * do the actual decoding
     */
	new_pkt = new gf* [k];
    for (row = 0 ; row < k ; row++ ) 
    {
		if (index[row] >= k) 
		{
		    new_pkt[row] = new gf [sz];
		    memset(new_pkt[row], 0, sz * sizeof(gf) ) ;
		    for (col = 0 ; col < k ; col++ )
		    {
				addmul(new_pkt[row], pkt[col], m_dec[row*k + col], sz) ;
			}
		}
    }
    /*
     * move pkts to their final destination
     */
    for (row = 0 ; row < k ; row++ ) 
    {
		if (index[row] >= k) 
		{
		    memcpy(pkt[row], new_pkt[row],  sz*sizeof(gf));
		    delete [] new_pkt[row];
		}
    }

    delete [] new_pkt;
    delete [] m_dec;

    // cout <<"fec_decode end "<<endl;

    return 0;
}




// static const int kSamplerate[4] = {16000,32000,48000,441000};
// static const int kFrameSize[4]  = {20 ,40,60,80};


RsFecEncoder::RsFecEncoder()
{
	m_k_ = FEC_K;
	src_pkg_cnt_ = 0;
    fec_pkg_idx_ = m_k_;
    max_size_ = 0;
    // sample_rate_id_ = 0;
    // frame_size_id_ = 0;
    // need_set_n_ = false;
	n_to_set_ = FEC_K + 1;
	m_n_ = FEC_K + 1;

	fec_flag_ = false;
    

    int i;
    for(i=0;i<MAX_FEC_N;i++)
    {
    	pkg_list_[i] = NULL;
    }

    for(i=0;i<MAX_FEC_N-FEC_K;i++)
    {
    	enc_mat_[i] = NULL;
    }
}

RsFecEncoder::~RsFecEncoder()
{
	int i;
	for(i=0;i<MAX_FEC_N;i++)
    {
    	if(pkg_list_[i] != NULL)
    	{
    		delete [] pkg_list_[i];
    	}
    }

    for(i=0;i<MAX_FEC_N-FEC_K;i++)
    {
    	if(enc_mat_[i] != NULL)
    	{
    		delete [] enc_mat_[i];
    	}
    }

}

int RsFecEncoder::Init(uint8_t n,int sample_rate)
{
	if(n<=MAX_FEC_N  && n>m_k_)
	{
		m_n_ = n;
		//m_k_ = k;
	}
	else
	{
		m_n_ = FEC_K + 1;
		//m_k_ = 4;
		
	}

	n_to_set_ = m_n_;
	// src_pkg_cnt_ = 0;
 //    fec_pkg_idx_ = m_k_;

    init_fec();
 //    if(enc_mat_[n-m_k_-1] == NULL)
	// {
	// 	enc_mat_[n-m_k_-1] = creat_encode_mat(m_n_,m_k_);
	// }

	SetSampleRate(sample_rate);
	//SetFrameSize(frame_size);
    return 0;
}


uint8_t* RsFecEncoder::GetEncodeMat(int n)
{
	if(n<=FEC_K)
	{
		return NULL;
	}

	if(enc_mat_[n-FEC_K-1] == NULL)
	{
		enc_mat_[n-FEC_K-1] = creat_encode_mat(n,FEC_K);
	}

	return enc_mat_[n-FEC_K-1];
}


bool RsFecEncoder::IsNeedSendRedPkg()
{
	if(m_n_ > 2*m_k_)
	{
		if(fec_pkg_idx_ < m_k_ || fec_pkg_idx_ >= m_n_ - m_k_ || pkg_list_[fec_pkg_idx_] == NULL || !fec_flag_)
		{
			return false;
		}
		else
		{
			return true;
		}
	}
	else
	{
		return false;
	}
}


bool RsFecEncoder::IsNeedAddRedData()
{
	if(m_n_ > m_k_)
	{
		if(fec_pkg_idx_ < m_k_ || pkg_list_[fec_pkg_idx_] == NULL || !fec_flag_)
		{
			return false;
		}
		else
		{
			return true;
		}
	}
	else
	{
		return false;
	}
}





int RsFecEncoder::CreateFecHeader(uint8_t *buf, int pt, int index,int pkg_len)
{
	uint32_t header = 0;

	int n = m_n_ ;

    if(fec_flag_)
    {
    	
// RS FEC headers :
//
//    0                   1                   2                   3
//    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//   |F|   block PT  |  timestamp offset         | block length  |     n   |  index  |
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
		header ^= 1 << 31;
	    header ^= pt << 24;
	    // header ^= frame_size_ << 10;
	    // header ^= pkg_len;
	    header ^= frame_size_ << 12;
	    header ^= pkg_len<<2;
	    header ^= n>>3;

	    for(int i=0;i<4;i++)
	    {
	    	buf[i] = (header >> (24-i*8)) & 0xff;
	    	// cout <<"buf[i]: "  << (int)buf[i] <<endl;
		}
		// buf[4] = (n << 4)^ index ;
		buf[4] = ((n&0x7) << 5)^ index ;
	    // cout <<"header: "  << header <<endl;
	    return 5;
	}
	else
	{
		buf[0] = pt;
		return 1;
	}
}


int RsFecEncoder::Encode(uint8_t *rtp_buf, int buf_len, const uint8_t *payload_data, int payload_size,int8_t pt,uint32_t ts)
{
	unsigned total_size = 0;
    //int8_t pt = pt_;

	if(rtp_buf == NULL || buf_len< 2*max_size_+4)
	{
		return -1;
	}

	if(payload_size == 0)
	{
		if(pkg_list_[fec_pkg_idx_] != NULL && fec_size_ > 0)
		{
			total_size = CreateFecHeader(rtp_buf,pt,fec_pkg_idx_,payload_size);
			memcpy(rtp_buf+total_size, pkg_list_[fec_pkg_idx_] , fec_size_);
			total_size += fec_size_;

			delete [] pkg_list_[fec_pkg_idx_];
        	pkg_list_[fec_pkg_idx_] = NULL;

			fec_pkg_idx_ ++;
			if(fec_pkg_idx_>=m_n_)
			{
				fec_pkg_idx_ = m_k_;
			}
		}
		else
		{
			return -1;
		}
	}
	else
	{
		if(payload_data == NULL)
		{
			return -1;
		}

		SetFrameSize(ts,payload_size);

		total_size = CreateFecHeader(rtp_buf,pt,src_pkg_cnt_,payload_size);

		memcpy(rtp_buf+total_size, payload_data , payload_size);

		total_size += payload_size;

		if(fec_flag_)
		{
			if(IsNeedAddRedData())
	        {
	        	memcpy(rtp_buf+total_size, pkg_list_[fec_pkg_idx_] , fec_size_);
	        	total_size += fec_size_;

	        	delete [] pkg_list_[fec_pkg_idx_];
	        	pkg_list_[fec_pkg_idx_] = NULL;

	        	fec_pkg_idx_ ++;
				if(fec_pkg_idx_ >= m_n_)
				{
					fec_pkg_idx_ = m_k_;
				}

	        }

			if(pkg_list_[src_pkg_cnt_] == NULL)
			{
				pkg_list_[src_pkg_cnt_] = new uint8_t[MAX_PAYLOAD_SIZE];
			}
	        
	        uint8_t *pbuf = pkg_list_[src_pkg_cnt_];
	        pbuf[0] = payload_size>>8;
	        pbuf[1] = payload_size&0xff;
	        pbuf += 2;
	        memcpy(pbuf, payload_data , payload_size);
	        memset(pbuf+payload_size, 0 , MAX_PAYLOAD_SIZE - payload_size-2);

	        // memcpy(pkg_list_[src_pkg_cnt_], payload_data , payload_size);
	        // memset(pkg_list_[src_pkg_cnt_]+payload_size, 0 , MAX_PAYLOAD_SIZE - payload_size);
	        if(payload_size > max_size_)
	        {
	        	max_size_ = payload_size;
	        }
	        
	        src_pkg_cnt_ ++;
	        if(src_pkg_cnt_ == m_k_)
	        {
	        	src_pkg_cnt_ = 0;
	        	fec_size_ = max_size_+2;
	        	max_size_ = 0;

	        	if(n_to_set_ != m_n_)
	        	{
	        		m_n_ = n_to_set_;
	        	}
	            
	            for(int i=m_k_;i<m_n_ ; i++)
	            {
	            	if(pkg_list_[i] != NULL)
					{
						delete [] pkg_list_[i];
					}
					pkg_list_[i] = new uint8_t[fec_size_];
	            	fec_encode(GetEncodeMat(m_n_), m_n_, m_k_, pkg_list_, pkg_list_[i], i, fec_size_);
	            }

	            for(int i=0;i<m_k_ ; i++)
	            {
	            	delete [] pkg_list_[i];
	            	pkg_list_[i] = NULL;
	            }
	        }
    	}
        
	}

	return total_size;
}




void RsFecEncoder::SetSampleRate(int sample_rate)
{
	sample_rate_ = sample_rate;

	//not support samplerate
}

int RsFecEncoder::GetSampleRate() const
{
	// return kSamplerate[sample_rate_id_];

	return sample_rate_;
}

void RsFecEncoder::SetFrameSize(uint32_t ts,int payload_size)
{
	int ts_offset = ts - ts_;

	if(fec_flag_)
	{
		if((ts != (ts_ + frame_size_)) || (payload_size <= 1))
		{
			fec_flag_ = false;
			for(int i=0;i<MAX_FEC_N;i++)
		    {
		    	if(pkg_list_[i] != NULL)
		    	{
		    		delete [] pkg_list_[i];
		    		pkg_list_[i] = NULL;
		    	}
		    }
			if(ts_offset>0 && ts_offset<sample_rate_/10)
			{
				fec_flag_ = true;
				src_pkg_cnt_ = 0;
			    fec_pkg_idx_ = m_k_;
			    max_size_ = 0;
				frame_size_ = ts_offset;
			}
			else
			{
				frame_size_ = 0;
			}
		}
	}
	else
	{
		if(ts_offset>0 && ts_offset<sample_rate_/10)
		{

			fec_flag_ = true;
			src_pkg_cnt_ = 0;
		    fec_pkg_idx_ = m_k_;
		    max_size_ = 0;

			frame_size_ = ts_offset;

		}
		else
		{
			frame_size_ = 0;
		}
	}

	ts_ = ts;
}

// int RsFecEncoder::GetFrameSize() const
// {
// 	return kFrameSize[frame_size_id_];
// }


void RsFecEncoder::SetN(uint8_t n)
{
	if(n> FEC_K && n<=MAX_FEC_N && m_n_ != n)
	{
		// m_n_ = n;
		// src_pkg_cnt_ = 0;
    	// fec_pkg_idx_ = m_k_;

    	// need_set_n_ = true;
    	n_to_set_ = n;

    	// if(enc_mat_[n-m_k_-1] == NULL)
    	// {
    	// 	enc_mat_[n-m_k_-1] = creat_encode_mat(n,m_k_);
    	// }

    	
	}
}

void RsFecEncoder::GetFecParameter(uint8_t &n, uint8_t &k)
{
	n = m_n_;
	k = m_k_; 
}


void RsFecEncoder::SetFECStatus(bool enable)
{
	enable_ = enable;
}

bool RsFecEncoder::GetFECStatus() const
{
	return enable_;
}

void RsFecEncoder::SetPt(int8_t pt)
{
	pt_ = pt;
}

int8_t RsFecEncoder::GetPt() const
{
	return pt_;
}





RsFecDecoder::RsFecDecoder()
{

	int i,k;
    for(i=0;i<MAX_LIST_NUM;i++)
    {
    	for(k=0;k<FEC_K;k++)
    	{
    		pkg_list_[i][k] = NULL;
    		pkg_index_[i][k] = -1;
    	}
    }

    for(i=0;i<MAX_FEC_N-FEC_K;i++)
    {
    	enc_mat_[i] = NULL;
    }

    for(i=0;i<MAX_LIST_NUM;i++)
    {
    	dec_para_[i] = NULL;
    }

    list_cnt_ = 0;
    decoded_sn_cnt_ = 0;
    m_n_ = 12;

    for(i=0;i<REG_SN_NUM;i++)
	{
		decoded_sn_[i] = 0;	
		decoded_ts_[i] = 0;	
	}
}

RsFecDecoder::~RsFecDecoder()
{
	int i,k;
    for(i=0;i<MAX_LIST_NUM;i++)
    {
    	for(k=0;k<FEC_K;k++)
    	{
    		if(pkg_list_[i][k] != NULL)
    		{
    			delete [] pkg_list_[i][k];
    		}
    	}
    }

    for(i=0;i<MAX_FEC_N-FEC_K;i++)
    {
    	if(enc_mat_[i] != NULL)
    	{
    		delete [] enc_mat_[i];
    	}
    }

    for(i=0;i<MAX_LIST_NUM;i++)
    {
    	if(dec_para_[i] != NULL)
    	{
    		delete [] dec_para_[i];
    	}
    }
}


int RsFecDecoder::Init()
{

    init_fec();

    return 0;
}

int RsFecDecoder::ParseFecHeader(const uint8_t *buf, uint8_t &pt,int &n,int &sid,int &rid,int &frame_size,int &src_size)
{
	uint32_t header = 0;
	int idx;

    for(int i=0;i<4;i++)
    {
		header ^= ((uint32_t)buf[i])<<((3-i)*8);
	}

	//cout <<"header: "  << header <<endl;

	pt = (header<<1)>>25;
	// n = (header>>(32-11))&(~((~0)<<4));
	// idx = (header>>(32-15))&(~((~0)<<4));
	//fid = (header>>10)&(~((~0)<<2));
	//idr = (header>>12)&(~((~0)<<2));
	// src_size = (header<<22)>>22;
	// frame_size = (header<<8)>>18;

	// n = (buf[4]>>4)+1;
	// idx = buf[4]&0x0f;
	frame_size = (header<<8)>>20;
	src_size = (header<<20)>>22;
	

	n = (buf[4]>>5)^((header&0x3)<<3);
	idx = buf[4]&0x1f;

	// cout <<"pt: "  << (int)pt <<",src_size: "  << src_size << ",frame_size: "  << frame_size <<",n: "  << n << ",idx: "  << idx <<endl;

    if(n<=FEC_K)
    {
    	return -1;
    }

	if(idx<FEC_K)
	{
		sid = idx;
		if(n>2*FEC_K)
		{
			rid = n-(FEC_K-idx);
		}
		else
		{
			rid = idx+FEC_K;
		}
	}
    else
    {
    	rid = idx;
    	sid = 0;

    	if(src_size!=0)
    	{
    		return -1;
    	}
    }

    m_n_ = n;
    // frame_size = kSamplerate[idr]*kFrameSize[fid]/1000;

    return 0;
}


uint8_t* RsFecDecoder::GetEncodeMat(int n)
{
	if(n<=FEC_K)
	{
		return NULL;
	}

	if(enc_mat_[n-FEC_K-1] == NULL)
	{
		enc_mat_[n-FEC_K-1] = creat_encode_mat(n,FEC_K);
	}

	return enc_mat_[n-FEC_K-1];
}

uint8_t RsFecDecoder::GetPt(const uint8_t *payload_buf)
{
	uint8_t pt = payload_buf[0]&0x7f;

	return pt;
}

uint32_t RsFecDecoder::GetLostRate()
{
	uint32_t ret=0;
	if(m_n_<25)
	{
		ret = ((m_n_-12)*5)>>1;
	}
	else
	{
		ret = (m_n_-21)*10;
	}

	if(ret>100)
	{
		return 100;
	}
	else
	{
		return ret;
	}
}

int RsFecDecoder::Decode(const uint8_t *payload_buf, int buf_size ,uint16_t sn , uint32_t ts, uint8_t *out_buf, int out_buf_len, uint8_t &out_pt)
{
	int n,sid,rid,frame_size,src_size;
	const uint8_t *pbuf;
	uint16_t base_sn ;
	uint32_t base_ts ;
	int header_size;
	uint8_t pt;

	if((payload_buf[0]>>7) == 0)
	{
		pt = payload_buf[0];
		header_size = 1;
		src_size = buf_size - header_size;
	}
	else
	{
 
	 	header_size = 5;

	    pbuf = payload_buf;
		if(ParseFecHeader(pbuf, pt,n,sid,rid,frame_size,src_size)<0)
		{
			// cout <<"decoder parse header error" <<endl;
			return -1;
		}

		pbuf += header_size;

		if(src_size == 0)
		{
			base_sn = sn - rid;
			base_ts = ts - (FEC_K-1)*frame_size;
			AddPkgToList(pbuf, buf_size - header_size,pt,n,rid,base_sn,base_ts,frame_size);
		}
		else
		{
			base_sn = sn - sid;
			base_ts = ts - sid*frame_size;
			AddPkgToList(pbuf, src_size,pt,n,sid,base_sn,base_ts,frame_size);

	        int red_size = buf_size - header_size - src_size;
	        if(red_size>0)
	        {
		        pbuf += src_size;
				base_sn = sn - rid;

				if(n>2*FEC_K)
				{
					base_ts = ts - (rid-(n-2*FEC_K))*frame_size;
				}
				else
				{
					base_ts = ts - rid*frame_size;
				}
				AddPkgToList(pbuf, red_size,pt,n,rid,base_sn,base_ts,frame_size);
			}
		}

		for(int i=0;i<MAX_LIST_NUM;i++)
	    {
	    	if(dec_para_[i]!=NULL)
	    	{
	    		if(dec_para_[i]->pkg_num >=FEC_K)
	    		{
	    			bool need_decode = false;

	    			decoded_sn_[decoded_sn_cnt_] = dec_para_[i]->sn;
	    			decoded_ts_[decoded_sn_cnt_] = dec_para_[i]->ts;
	    			decoded_sn_cnt_ ++;
	    			if(decoded_sn_cnt_ >= REG_SN_NUM)
	    			{
	    				decoded_sn_cnt_ = 0;
	    			}

	    			for(int k=0;k<FEC_K;k++)
	    			{
	    				if(pkg_index_[i][k] >=  FEC_K)
	    				{
	    					need_decode = true;
	    					break ;
	    				}
	    			}

	    			if(!dec_para_[i]->is_decoded && need_decode)
	    			{
		    			if(fec_decode(GetEncodeMat(dec_para_[i]->n), dec_para_[i]->n, FEC_K, pkg_list_[i], pkg_index_[i], dec_para_[i]->size)== 0)
		    			{
		    				dec_para_[i]->is_decoded = true;
		    			}
		    			else
		    			{
		    				DeletePkgList(i);
		    			}
	    			}
	    			else
	    			{
	    				DeletePkgList(i);
	    			}


	    		}
	    	}
	    }
	}
    
    if(src_size>out_buf_len)
    {
    	return -1;
    }
    else
    {
    	out_pt = pt;
    	memcpy(out_buf,payload_buf+header_size,src_size);
    	return src_size;
    }
}


int RsFecDecoder::AddPkgToList(const uint8_t *buf, int buf_size,uint8_t pt,int n,int id,uint16_t base_sn,uint32_t base_ts,int frame_size)
{
	int i,k;

	if(buf_size<=1)
	{
		return -1;
	}

	for(i=0;i<REG_SN_NUM;i++)
	{
		//if(base_sn == decoded_sn_[i])
		if(base_ts == decoded_ts_[i])
		{
			return 0;
		}
	}
    
    for(i=0;i<MAX_LIST_NUM;i++)
    {
    	if(dec_para_[i]!=NULL)
    	{
    		//if(dec_para_[i]->sn == base_sn)
    		if(dec_para_[i]->ts == base_ts)
    		{
    			for(k=0;k<FEC_K;k++)
    			{
    				if(pkg_index_[i][k] == id)
    				{
    					return 0;
    				}
    			}
    			if(dec_para_[i]->pkg_num < FEC_K)
    			{
    				pkg_list_[i][dec_para_[i]->pkg_num] = new uint8_t[MAX_PAYLOAD_SIZE];
    				if(id<FEC_K)
    				{
    				    uint8_t *pbuf = pkg_list_[i][dec_para_[i]->pkg_num];
				        pbuf[0] = buf_size>>8;
				        pbuf[1] = buf_size&0xff;
				        pbuf += 2;
				        memcpy(pbuf, buf , buf_size);
				        memset(pbuf+buf_size, 0 , MAX_PAYLOAD_SIZE - buf_size-2);
    				}
    				else
    				{
	    				memcpy(pkg_list_[i][dec_para_[i]->pkg_num] , buf, buf_size);
	    				memset(pkg_list_[i][dec_para_[i]->pkg_num]+buf_size,0,MAX_PAYLOAD_SIZE-buf_size);
    				}

    				pkg_index_[i][dec_para_[i]->pkg_num] = id;

    				dec_para_[i]->pkg_num++;
    				if(id >= FEC_K)
    				{
    					dec_para_[i]->size = buf_size;
    					dec_para_[i]->n = n;
    				}
    			}
    			return 0;
    		}
    	}
    }

    if(dec_para_[list_cnt_]==NULL)
    {
    	dec_para_[list_cnt_] = new RsDecoderPara;
    }
    else
    {
    	//LOG(INFO)<< "RsFecDecoder::AddPkgToList delete list,sn="<< dec_para_[list_cnt_]->sn << ",list_cnt_="<< list_cnt_;
    	for(k=0;k<FEC_K;k++)
    	{
    		if(pkg_list_[list_cnt_][k] != NULL)
    		{
    			delete [] pkg_list_[list_cnt_][k];
    			pkg_list_[list_cnt_][k] = NULL;
    		}
    	}
    }
	
	//LOG(INFO)<< "AddPkgToList ,sn="<< base_sn<< ",ts="<< base_ts << ",id="<< id<< ",list_cnt_="<< list_cnt_;

	for(k=0;k<FEC_K;k++)
	{
		pkg_index_[list_cnt_][k] = -1;
	}
	
	pkg_list_[list_cnt_][0] = new uint8_t[MAX_PAYLOAD_SIZE];
	if(id<FEC_K)
	{
	    uint8_t *pbuf = pkg_list_[list_cnt_][0];
        pbuf[0] = buf_size>>8;
        pbuf[1] = buf_size&0xff;
        pbuf += 2;
        memcpy(pbuf, buf , buf_size);
        memset(pbuf+buf_size, 0 , MAX_PAYLOAD_SIZE - buf_size-2);
        dec_para_[list_cnt_]->size = buf_size+2;
	}
	else
	{
		memcpy(pkg_list_[list_cnt_][0] , buf, buf_size);
		memset(pkg_list_[list_cnt_][0]+buf_size,0,MAX_PAYLOAD_SIZE-buf_size);
		dec_para_[list_cnt_]->size = buf_size;
	}

	pkg_index_[list_cnt_][0] = id;

	dec_para_[list_cnt_]->pkg_num = 1;
	dec_para_[list_cnt_]->n = n;
	dec_para_[list_cnt_]->pt = pt;
	dec_para_[list_cnt_]->sn = base_sn;
	dec_para_[list_cnt_]->ts = base_ts;
	//dec_para_[list_cnt_]->size = buf_size;
	dec_para_[list_cnt_]->is_decoded = false;
	dec_para_[list_cnt_]->pkg_rd_cnt = 0;
	dec_para_[list_cnt_]->frame_size = frame_size;

	list_cnt_++;
	if(list_cnt_>=MAX_LIST_NUM)
	{
		list_cnt_ = 0;
	}

	return 0;

}



int RsFecDecoder::GetRecoveryData(uint8_t *buf, int buf_len, uint16_t &sn , uint32_t &ts, uint8_t &pt)
{
	for(int i=0;i<MAX_LIST_NUM;i++)
    {
    	if(dec_para_[i]!=NULL)
    	{
    		if(dec_para_[i]->is_decoded )
    		{
    			while(dec_para_[i]->pkg_rd_cnt < FEC_K)
    			{
	    			if(pkg_index_[i][dec_para_[i]->pkg_rd_cnt]>=FEC_K)
	    			{
	    				if(buf_len>=dec_para_[i]->size)
	    				{
	    					uint8_t *pbuf = pkg_list_[i][dec_para_[i]->pkg_rd_cnt];
	    					int pkg_len = (pbuf[0]<<8) ^ pbuf[1];
	    					int size;
	    					if(pkg_len<dec_para_[i]->size)
	    					{
	    						size = pkg_len;
	    					} 
	    					else
	    					{
	    						size = dec_para_[i]->size-2;
	    					}
	    					memcpy(buf,pbuf+2,size);
	    					delete [] pkg_list_[i][dec_para_[i]->pkg_rd_cnt];
	    					pkg_list_[i][dec_para_[i]->pkg_rd_cnt] = NULL;
	    					sn = dec_para_[i]->sn + dec_para_[i]->pkg_rd_cnt;
	    					ts = dec_para_[i]->ts + dec_para_[i]->pkg_rd_cnt*dec_para_[i]->frame_size;
	    					pt = dec_para_[i]->pt;

	    					// cout <<"dec_para_ ts: "  << dec_para_[i]->ts <<",frame_size: "  << dec_para_[i]->frame_size << ",ts: "  << ts<<endl;

	    					// int size = dec_para_[i]->size;
	    					dec_para_[i]->pkg_rd_cnt ++;
	    					if(dec_para_[i]->pkg_rd_cnt >= FEC_K)
	    					{
								DeletePkgList(i);
	    					}

	    					return size;
	    				}
	    				else
	    				{
	    					return -1;
	    				}
	    			}

	    			dec_para_[i]->pkg_rd_cnt ++;
    			}
				DeletePkgList(i);
    		}
    	}
    }

    return 0;
}


void RsFecDecoder::DeletePkgList(int index)
{
	if(dec_para_[index] != NULL)
	{
		delete dec_para_[index];
		dec_para_[index] = NULL;
	}

	for(int k=0;k<FEC_K;k++)
	{
		if(pkg_list_[index][k] != NULL)
		{
			delete [] pkg_list_[index][k];
			pkg_list_[index][k] = NULL;
		}
	} 
}