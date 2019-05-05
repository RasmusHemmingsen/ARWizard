using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ZombieController : MonoBehaviour
{
    Animator animator;

    private void Awake()
    {
        animator = GetComponent<Animator>();
    }

    private void OnTriggerEnter(Collider other)
    {
        ZombieHit();
    }

    public void ZombieHit()
    {
        animator.SetTrigger("OnHit");
    }

    private void Resurrect()
    {
        animator.ResetTrigger("OnHit");
        animator.SetTrigger("Resurrect");
    }

}